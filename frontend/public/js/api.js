/**
 * Corruption Reporting System - Backend API Wrapper
 * Version: 1.0.0
 * Description: Clean interface for all backend API communications
 * 
 * This module provides methods for:
 * - Evidence submission
 * - Submission retrieval
 * - Counter-evidence submission
 * - Report generation
 * - Health checks
 * 
 * Usage:
 *   import { CorruptionReportingAPI } from './api.js';
 *   const api = new CorruptionReportingAPI();
 *   const result = await api.submitEvidence(formData);
 */

// ============================================
// CONFIGURATION
// ============================================
const API_CONFIG = {
    baseURL: window.location.origin,
    apiPrefix: '/api/v1',
    timeout: 300000, // 5 minutes for ML processing
    retryAttempts: 0,
    retryDelay: 1000
};

// ============================================
// ERROR CLASSES
// ============================================

/**
 * Custom API Error class
 */
class APIError extends Error {
    constructor(message, status, code, details) {
        super(message);
        this.name = 'APIError';
        this.status = status;
        this.code = code;
        this.details = details;
        this.timestamp = new Date().toISOString();
    }
}

/**
 * Network Error class
 */
class NetworkError extends Error {
    constructor(message, originalError) {
        super(message);
        this.name = 'NetworkError';
        this.originalError = originalError;
        this.timestamp = new Date().toISOString();
    }
}

/**
 * Timeout Error class
 */
class TimeoutError extends Error {
    constructor(message, timeout) {
        super(message);
        this.name = 'TimeoutError';
        this.timeout = timeout;
        this.timestamp = new Date().toISOString();
    }
}

// ============================================
// API CLIENT CLASS
// ============================================

class CorruptionReportingAPI {
    constructor(config = {}) {
        this.config = { ...API_CONFIG, ...config };
        this.baseURL = this.config.baseURL;
        this.apiPrefix = this.config.apiPrefix;
        this.timeout = this.config.timeout;
        
        // Track active requests
        this.activeRequests = new Map();
        
        // Request ID counter
        this.requestIdCounter = 0;
    }

    // ============================================
    // CORE HTTP METHODS
    // ============================================

    /**
     * Generate unique request ID
     * @returns {string} Request ID
     */
    _generateRequestId() {
        return `req_${Date.now()}_${++this.requestIdCounter}`;
    }

    /**
     * Build full URL
     * @param {string} endpoint - API endpoint
     * @returns {string} Full URL
     */
    _buildURL(endpoint) {
        return `${this.baseURL}${this.apiPrefix}${endpoint}`;
    }

    /**
     * Create abort controller with timeout
     * @param {number} timeout - Timeout in milliseconds
     * @returns {AbortController} Abort controller
     */
    _createAbortController(timeout) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => {
            controller.abort();
        }, timeout);
        
        // Store timeout ID to clear it later
        controller.timeoutId = timeoutId;
        
        return controller;
    }

    /**
     * Handle fetch response
     * @param {Response} response - Fetch response
     * @returns {Promise<any>} Parsed response data
     */
    async _handleResponse(response) {
        const contentType = response.headers.get('content-type');
        
        // Handle different content types
        if (contentType && contentType.includes('application/json')) {
            const data = await response.json();
            
            if (!response.ok) {
                throw new APIError(
                    data.error || data.message || 'API request failed',
                    response.status,
                    data.code || 'UNKNOWN_ERROR',
                    data.details || data
                );
            }
            
            return data;
        } else if (contentType && contentType.includes('application/pdf')) {
            if (!response.ok) {
                throw new APIError(
                    'Failed to download PDF report',
                    response.status,
                    'PDF_DOWNLOAD_ERROR',
                    null
                );
            }
            
            return await response.blob();
        } else {
            if (!response.ok) {
                const text = await response.text();
                throw new APIError(
                    text || 'API request failed',
                    response.status,
                    'UNKNOWN_ERROR',
                    null
                );
            }
            
            return await response.text();
        }
    }

    /**
     * Handle fetch errors
     * @param {Error} error - Error object
     * @param {AbortController} controller - Abort controller
     * @throws {APIError|NetworkError|TimeoutError}
     */
    _handleError(error, controller) {
        // Clear timeout
        if (controller && controller.timeoutId) {
            clearTimeout(controller.timeoutId);
        }
        
        // Already an API error
        if (error instanceof APIError) {
            throw error;
        }
        
        // Timeout error
        if (error.name === 'AbortError') {
            throw new TimeoutError(
                `Request timed out after ${this.timeout}ms`,
                this.timeout
            );
        }
        
        // Network error
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            throw new NetworkError(
                'Network request failed. Please check your connection.',
                error
            );
        }
        
        // Generic error
        throw new NetworkError(error.message, error);
    }

    /**
     * Make HTTP request
     * @param {string} endpoint - API endpoint
     * @param {Object} options - Fetch options
     * @returns {Promise<any>} Response data
     */
    async _request(endpoint, options = {}) {
        const requestId = this._generateRequestId();
        const url = this._buildURL(endpoint);
        const controller = this._createAbortController(options.timeout || this.timeout);
        
        // Default headers
        const headers = {
            ...options.headers
        };
        
        // Add JSON content-type if body is present and not FormData
        if (options.body && !(options.body instanceof FormData)) {
            headers['Content-Type'] = 'application/json';
        }
        
        const fetchOptions = {
            ...options,
            headers,
            signal: controller.signal
        };
        
        // Track active request
        this.activeRequests.set(requestId, controller);
        
        try {
            const response = await fetch(url, fetchOptions);
            const data = await this._handleResponse(response);
            
            return data;
        } catch (error) {
            this._handleError(error, controller);
        } finally {
            // Clean up
            if (controller.timeoutId) {
                clearTimeout(controller.timeoutId);
            }
            this.activeRequests.delete(requestId);
        }
    }

    /**
     * GET request
     * @param {string} endpoint - API endpoint
     * @param {Object} params - Query parameters
     * @returns {Promise<any>} Response data
     */
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const fullEndpoint = queryString ? `${endpoint}?${queryString}` : endpoint;
        
        return this._request(fullEndpoint, {
            method: 'GET'
        });
    }

    /**
     * POST request
     * @param {string} endpoint - API endpoint
     * @param {Object|FormData} data - Request body
     * @returns {Promise<any>} Response data
     */
    async post(endpoint, data) {
        const body = data instanceof FormData ? data : JSON.stringify(data);
        
        return this._request(endpoint, {
            method: 'POST',
            body
        });
    }

    /**
     * PUT request
     * @param {string} endpoint - API endpoint
     * @param {Object} data - Request body
     * @returns {Promise<any>} Response data
     */
    async put(endpoint, data) {
        return this._request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }

    /**
     * DELETE request
     * @param {string} endpoint - API endpoint
     * @returns {Promise<any>} Response data
     */
    async delete(endpoint) {
        return this._request(endpoint, {
            method: 'DELETE'
        });
    }

    // ============================================
    // HEALTH CHECK
    // ============================================

    /**
     * Check API health
     * @returns {Promise<Object>} Health status
     */
    async checkHealth() {
        return this.get('/health');
    }

    // ============================================
    // SUBMISSION ENDPOINTS
    // ============================================

    /**
     * Submit evidence
     * @param {FormData} formData - Form data with evidence
     * @returns {Promise<Object>} Submission response
     * 
     * Expected FormData fields:
     * - text: string (required) - Evidence description
     * - location: string (optional) - Location of incident
     * - images: File[] (optional) - Image evidence
     * - audio: File[] (optional) - Audio evidence
     * - video: File[] (optional) - Video evidence
     * - documents: File[] (optional) - Document evidence
     */
    async submitEvidence(formData) {
        if (!(formData instanceof FormData)) {
            throw new Error('submitEvidence requires FormData');
        }
        
        return this.post('/submissions', formData);
    }

    /**
     * Get all submissions
     * @param {Object} options - Query options
     * @param {number} options.limit - Maximum number of results
     * @param {number} options.offset - Offset for pagination
     * @param {string} options.status - Filter by status
     * @returns {Promise<Object>} List of submissions
     */
    async getSubmissions(options = {}) {
        return this.get('/submissions', options);
    }

    /**
     * Get submission by ID
     * @param {string} submissionId - Submission ID
     * @returns {Promise<Object>} Submission details
     */
    async getSubmission(submissionId) {
        if (!submissionId) {
            throw new Error('Submission ID is required');
        }
        
        return this.get(`/submissions/${submissionId}`);
    }

    /**
     * Poll submission status
     * @param {string} submissionId - Submission ID
     * @param {Object} options - Polling options
     * @param {number} options.interval - Polling interval in ms (default: 2000)
     * @param {number} options.maxAttempts - Maximum polling attempts (default: 150)
     * @param {Function} options.onUpdate - Callback on each update
     * @returns {Promise<Object>} Final submission data
     */
    async pollSubmission(submissionId, options = {}) {
        const interval = options.interval || 2000;
        const maxAttempts = options.maxAttempts || 150; // 5 minutes max
        const onUpdate = options.onUpdate || (() => {});
        
        let attempts = 0;
        
        return new Promise((resolve, reject) => {
            const poll = async () => {
                try {
                    const submission = await this.getSubmission(submissionId);
                    
                    // Call update callback
                    onUpdate(submission);
                    
                    // Check if processing is complete
                    if (submission.status === 'completed' || submission.status === 'failed') {
                        resolve(submission);
                        return;
                    }
                    
                    // Check max attempts
                    attempts++;
                    if (attempts >= maxAttempts) {
                        reject(new TimeoutError(
                            'Polling timed out. Submission is still processing.',
                            maxAttempts * interval
                        ));
                        return;
                    }
                    
                    // Schedule next poll
                    setTimeout(poll, interval);
                } catch (error) {
                    reject(error);
                }
            };
            
            // Start polling
            poll();
        });
    }

    // ============================================
    // COUNTER-EVIDENCE ENDPOINTS
    // ============================================

    /**
     * Submit counter-evidence
     * @param {Object} data - Counter-evidence data
     * @param {string} data.submission_id - Target submission ID
     * @param {string} data.text - Counter-evidence text
     * @param {boolean} data.is_verified - Whether submitter is verified
     * @param {File[]} data.evidence_files - Supporting files (optional)
     * @returns {Promise<Object>} Counter-evidence response
     */
    async submitCounterEvidence(data) {
        if (!data.submission_id) {
            throw new Error('submission_id is required');
        }
        if (!data.text) {
            throw new Error('text is required');
        }
        
        // If evidence files are provided, use FormData
        if (data.evidence_files && data.evidence_files.length > 0) {
            const formData = new FormData();
            formData.append('submission_id', data.submission_id);
            formData.append('text', data.text);
            formData.append('is_verified', data.is_verified || false);
            
            // Add files
            data.evidence_files.forEach(file => {
                formData.append('evidence_files', file);
            });
            
            return this.post('/counter-evidence', formData);
        }
        
        // Otherwise use JSON
        return this.post('/counter-evidence', {
            submission_id: data.submission_id,
            text: data.text,
            is_verified: data.is_verified || false
        });
    }

    // ============================================
    // REPORT ENDPOINTS
    // ============================================

    /**
     * Download report PDF
     * @param {string} submissionId - Submission ID
     * @returns {Promise<Blob>} PDF blob
     */
    async downloadReport(submissionId) {
        if (!submissionId) {
            throw new Error('Submission ID is required');
        }
        
        return this.get(`/reports/${submissionId}`);
    }

    /**
     * Download report and trigger browser download
     * @param {string} submissionId - Submission ID
     * @param {string} filename - Filename for download (optional)
     * @returns {Promise<void>}
     */
    async downloadReportFile(submissionId, filename) {
        const blob = await this.downloadReport(submissionId);
        
        // Create download link
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename || `report_${submissionId}.pdf`;
        
        // Trigger download
        document.body.appendChild(link);
        link.click();
        
        // Cleanup
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }

    // ============================================
    // UTILITY METHODS
    // ============================================

    /**
     * Cancel all active requests
     */
    cancelAllRequests() {
        this.activeRequests.forEach((controller, requestId) => {
            controller.abort();
            if (controller.timeoutId) {
                clearTimeout(controller.timeoutId);
            }
        });
        this.activeRequests.clear();
    }

    /**
     * Get number of active requests
     * @returns {number} Number of active requests
     */
    getActiveRequestCount() {
        return this.activeRequests.size;
    }

    /**
     * Check if API is available
     * @returns {Promise<boolean>} True if API is available
     */
    async isAvailable() {
        try {
            await this.checkHealth();
            return true;
        } catch (error) {
            return false;
        }
    }
}

// ============================================
// SINGLETON INSTANCE
// ============================================

// Create default instance
const api = new CorruptionReportingAPI();

// ============================================
// CONVENIENCE FUNCTIONS
// ============================================

/**
 * Submit evidence (convenience function)
 * @param {FormData} formData - Form data
 * @returns {Promise<Object>} Submission response
 */
async function submitEvidence(formData) {
    return api.submitEvidence(formData);
}

/**
 * Get submission by ID (convenience function)
 * @param {string} submissionId - Submission ID
 * @returns {Promise<Object>} Submission details
 */
async function getSubmission(submissionId) {
    return api.getSubmission(submissionId);
}

/**
 * Submit counter-evidence (convenience function)
 * @param {Object} data - Counter-evidence data
 * @returns {Promise<Object>} Counter-evidence response
 */
async function submitCounterEvidence(data) {
    return api.submitCounterEvidence(data);
}

/**
 * Download report (convenience function)
 * @param {string} submissionId - Submission ID
 * @param {string} filename - Filename
 * @returns {Promise<void>}
 */
async function downloadReport(submissionId, filename) {
    return api.downloadReportFile(submissionId, filename);
}

/**
 * Check API health (convenience function)
 * @returns {Promise<Object>} Health status
 */
async function checkHealth() {
    return api.checkHealth();
}

// ============================================
// ERROR HELPERS
// ============================================

/**
 * Check if error is an API error
 * @param {Error} error - Error object
 * @returns {boolean} True if API error
 */
function isAPIError(error) {
    return error instanceof APIError;
}

/**
 * Check if error is a network error
 * @param {Error} error - Error object
 * @returns {boolean} True if network error
 */
function isNetworkError(error) {
    return error instanceof NetworkError;
}

/**
 * Check if error is a timeout error
 * @param {Error} error - Error object
 * @returns {boolean} True if timeout error
 */
function isTimeoutError(error) {
    return error instanceof TimeoutError;
}

/**
 * Get user-friendly error message
 * @param {Error} error - Error object
 * @returns {string} User-friendly message
 */
function getErrorMessage(error) {
    if (isAPIError(error)) {
        return error.message;
    }
    
    if (isNetworkError(error)) {
        return 'Network error. Please check your connection and try again.';
    }
    
    if (isTimeoutError(error)) {
        return 'Request timed out. The server took too long to respond.';
    }
    
    return error.message || 'An unexpected error occurred.';
}

/**
 * Get error details for debugging
 * @param {Error} error - Error object
 * @returns {Object} Error details
 */
function getErrorDetails(error) {
    return {
        name: error.name,
        message: error.message,
        status: error.status,
        code: error.code,
        details: error.details,
        timestamp: error.timestamp,
        stack: error.stack
    };
}

// ============================================
// EXPORTS
// ============================================

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CorruptionReportingAPI,
        APIError,
        NetworkError,
        TimeoutError,
        api,
        submitEvidence,
        getSubmission,
        submitCounterEvidence,
        downloadReport,
        checkHealth,
        isAPIError,
        isNetworkError,
        isTimeoutError,
        getErrorMessage,
        getErrorDetails
    };
}

// Export for browser (global)
if (typeof window !== 'undefined') {
    window.CorruptionReportingAPI = CorruptionReportingAPI;
    window.api = api;
    window.APIError = APIError;
    window.NetworkError = NetworkError;
    window.TimeoutError = TimeoutError;
    
    // Convenience functions
    window.submitEvidence = submitEvidence;
    window.getSubmission = getSubmission;
    window.submitCounterEvidence = submitCounterEvidence;
    window.downloadReport = downloadReport;
    window.checkHealth = checkHealth;
    
    // Error helpers
    window.isAPIError = isAPIError;
    window.isNetworkError = isNetworkError;
    window.isTimeoutError = isTimeoutError;
    window.getErrorMessage = getErrorMessage;
    window.getErrorDetails = getErrorDetails;
}
