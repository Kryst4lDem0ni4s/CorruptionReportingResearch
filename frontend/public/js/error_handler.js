/**
 * Corruption Reporting System - Error Handler
 * Version: 1.0.0
 * Description: Global error handling and user notification system
 * 
 * This module provides:
 * - Global error handling
 * - Toast notifications
 * - Error logging
 * - User-friendly error messages
 * - Error recovery suggestions
 * - Network error detection
 * 
 * Dependencies: utils.js (optional)
 */

// ============================================
// CONFIGURATION
// ============================================
const ERROR_CONFIG = {
    toastDuration: 5000, // 5 seconds
    maxToasts: 3,
    logErrors: true,
    showStackTrace: false, // Show in development only
    retryAttempts: 3
};

// ============================================
// ERROR TYPES
// ============================================
const ERROR_TYPES = {
    NETWORK: 'network',
    API: 'api',
    VALIDATION: 'validation',
    AUTHENTICATION: 'authentication',
    AUTHORIZATION: 'authorization',
    NOT_FOUND: 'not_found',
    SERVER: 'server',
    CLIENT: 'client',
    TIMEOUT: 'timeout',
    UNKNOWN: 'unknown'
};

// ============================================
// ERROR SEVERITY
// ============================================
const ERROR_SEVERITY = {
    INFO: 'info',
    WARNING: 'warning',
    ERROR: 'error',
    CRITICAL: 'critical'
};

// ============================================
// TOAST NOTIFICATION SYSTEM
// ============================================

let toastContainer = null;
let activeToasts = [];

/**
 * Initialize toast container
 */
function initToastContainer() {
    if (toastContainer) return;
    
    toastContainer = document.createElement('div');
    toastContainer.id = 'toast-container';
    toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
    toastContainer.style.zIndex = '9999';
    document.body.appendChild(toastContainer);
}

/**
 * Show toast notification
 * @param {string} message - Toast message
 * @param {string} type - Toast type (success, error, warning, info)
 * @param {number} duration - Display duration in ms
 */
function showToast(message, type = 'info', duration = ERROR_CONFIG.toastDuration) {
    initToastContainer();
    
    // Remove oldest toast if limit reached
    if (activeToasts.length >= ERROR_CONFIG.maxToasts) {
        const oldestToast = activeToasts.shift();
        if (oldestToast && oldestToast.parentNode) {
            oldestToast.parentNode.removeChild(oldestToast);
        }
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${getBootstrapColor(type)} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    // Toast content
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${getIcon(type)} ${escapeHtml(message)}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;
    
    // Add to container
    toastContainer.appendChild(toast);
    activeToasts.push(toast);
    
    // Show toast (Bootstrap)
    if (typeof bootstrap !== 'undefined' && bootstrap.Toast) {
        const bsToast = new bootstrap.Toast(toast, {
            autohide: true,
            delay: duration
        });
        bsToast.show();
        
        // Remove from active list when hidden
        toast.addEventListener('hidden.bs.toast', () => {
            const index = activeToasts.indexOf(toast);
            if (index > -1) {
                activeToasts.splice(index, 1);
            }
            if (toast.parentNode) {
                toast.parentNode.removeChild(toast);
            }
        });
    } else {
        // Fallback without Bootstrap
        toast.style.display = 'block';
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => {
                const index = activeToasts.indexOf(toast);
                if (index > -1) {
                    activeToasts.splice(index, 1);
                }
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }, duration);
    }
}

/**
 * Get Bootstrap color class
 * @param {string} type - Toast type
 * @returns {string} Bootstrap color class
 */
function getBootstrapColor(type) {
    const colors = {
        success: 'success',
        error: 'danger',
        warning: 'warning',
        info: 'info'
    };
    return colors[type] || 'info';
}

/**
 * Get icon for toast type
 * @param {string} type - Toast type
 * @returns {string} Icon HTML
 */
function getIcon(type) {
    const icons = {
        success: '<i class="bi bi-check-circle me-2"></i>',
        error: '<i class="bi bi-exclamation-triangle me-2"></i>',
        warning: '<i class="bi bi-exclamation-circle me-2"></i>',
        info: '<i class="bi bi-info-circle me-2"></i>'
    };
    return icons[type] || '';
}

/**
 * Escape HTML
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

// ============================================
// ERROR CLASSIFICATION
// ============================================

/**
 * Classify error type
 * @param {Error} error - Error object
 * @returns {string} Error type
 */
function classifyError(error) {
    if (!error) return ERROR_TYPES.UNKNOWN;
    
    // Network errors
    if (error.name === 'NetworkError' || error.message.includes('network') || error.message.includes('fetch')) {
        return ERROR_TYPES.NETWORK;
    }
    
    // Timeout errors
    if (error.name === 'TimeoutError' || error.message.includes('timeout')) {
        return ERROR_TYPES.TIMEOUT;
    }
    
    // API errors
    if (error.name === 'APIError') {
        if (error.status === 401) return ERROR_TYPES.AUTHENTICATION;
        if (error.status === 403) return ERROR_TYPES.AUTHORIZATION;
        if (error.status === 404) return ERROR_TYPES.NOT_FOUND;
        if (error.status >= 500) return ERROR_TYPES.SERVER;
        if (error.status >= 400) return ERROR_TYPES.VALIDATION;
        return ERROR_TYPES.API;
    }
    
    return ERROR_TYPES.CLIENT;
}

/**
 * Get error severity
 * @param {string} errorType - Error type
 * @returns {string} Error severity
 */
function getErrorSeverity(errorType) {
    const severityMap = {
        [ERROR_TYPES.NETWORK]: ERROR_SEVERITY.WARNING,
        [ERROR_TYPES.TIMEOUT]: ERROR_SEVERITY.WARNING,
        [ERROR_TYPES.VALIDATION]: ERROR_SEVERITY.INFO,
        [ERROR_TYPES.NOT_FOUND]: ERROR_SEVERITY.WARNING,
        [ERROR_TYPES.AUTHENTICATION]: ERROR_SEVERITY.ERROR,
        [ERROR_TYPES.AUTHORIZATION]: ERROR_SEVERITY.ERROR,
        [ERROR_TYPES.SERVER]: ERROR_SEVERITY.CRITICAL,
        [ERROR_TYPES.API]: ERROR_SEVERITY.ERROR,
        [ERROR_TYPES.CLIENT]: ERROR_SEVERITY.ERROR,
        [ERROR_TYPES.UNKNOWN]: ERROR_SEVERITY.ERROR
    };
    return severityMap[errorType] || ERROR_SEVERITY.ERROR;
}

// ============================================
// USER-FRIENDLY MESSAGES
// ============================================

/**
 * Get user-friendly error message
 * @param {Error} error - Error object
 * @returns {string} User-friendly message
 */
function getUserFriendlyMessage(error) {
    const errorType = classifyError(error);
    
    const messages = {
        [ERROR_TYPES.NETWORK]: 'Network connection lost. Please check your internet connection and try again.',
        [ERROR_TYPES.TIMEOUT]: 'Request timed out. The server took too long to respond. Please try again.',
        [ERROR_TYPES.VALIDATION]: error.message || 'Invalid input. Please check your data and try again.',
        [ERROR_TYPES.AUTHENTICATION]: 'Authentication failed. Please log in again.',
        [ERROR_TYPES.AUTHORIZATION]: 'You do not have permission to perform this action.',
        [ERROR_TYPES.NOT_FOUND]: 'The requested resource was not found.',
        [ERROR_TYPES.SERVER]: 'Server error occurred. Please try again later.',
        [ERROR_TYPES.API]: error.message || 'An error occurred while processing your request.',
        [ERROR_TYPES.CLIENT]: error.message || 'An unexpected error occurred.',
        [ERROR_TYPES.UNKNOWN]: 'An unexpected error occurred. Please try again.'
    };
    
    return messages[errorType] || error.message || 'An error occurred.';
}

/**
 * Get recovery suggestions
 * @param {Error} error - Error object
 * @returns {Array<string>} Recovery suggestions
 */
function getRecoverySuggestions(error) {
    const errorType = classifyError(error);
    
    const suggestions = {
        [ERROR_TYPES.NETWORK]: [
            'Check your internet connection',
            'Try refreshing the page',
            'Disable VPN if enabled'
        ],
        [ERROR_TYPES.TIMEOUT]: [
            'Try again in a few moments',
            'Check your internet speed',
            'The server may be experiencing high load'
        ],
        [ERROR_TYPES.VALIDATION]: [
            'Review the form for errors',
            'Ensure all required fields are filled',
            'Check that file sizes are within limits'
        ],
        [ERROR_TYPES.AUTHENTICATION]: [
            'Clear your browser cache and cookies',
            'Try logging in again',
            'Contact support if the problem persists'
        ],
        [ERROR_TYPES.AUTHORIZATION]: [
            'Verify you have the necessary permissions',
            'Contact an administrator',
            'Log in with a different account'
        ],
        [ERROR_TYPES.NOT_FOUND]: [
            'Check the URL',
            'The resource may have been moved or deleted',
            'Return to the homepage'
        ],
        [ERROR_TYPES.SERVER]: [
            'Try again in a few minutes',
            'Contact support if the problem persists',
            'Check the status page for outages'
        ],
        [ERROR_TYPES.API]: [
            'Try again',
            'Refresh the page',
            'Contact support if the error continues'
        ]
    };
    
    return suggestions[errorType] || [
        'Try refreshing the page',
        'Clear your browser cache',
        'Contact support if the problem persists'
    ];
}

// ============================================
// ERROR DISPLAY
// ============================================

/**
 * Display error to user
 * @param {Error} error - Error object
 * @param {Object} options - Display options
 */
function displayError(error, options = {}) {
    const message = getUserFriendlyMessage(error);
    const errorType = classifyError(error);
    const severity = getErrorSeverity(errorType);
    
    // Determine toast type
    let toastType = 'error';
    if (severity === ERROR_SEVERITY.WARNING) toastType = 'warning';
    if (severity === ERROR_SEVERITY.INFO) toastType = 'info';
    
    // Show toast
    if (options.useToast !== false) {
        showToast(message, toastType);
    }
    
    // Show in error container if specified
    if (options.containerId) {
        showErrorInContainer(options.containerId, error);
    }
    
    // Log error
    if (ERROR_CONFIG.logErrors) {
        logError(error);
    }
}

/**
 * Show error in container
 * @param {string} containerId - Container element ID
 * @param {Error} error - Error object
 */
function showErrorInContainer(containerId, error) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    const message = getUserFriendlyMessage(error);
    const suggestions = getRecoverySuggestions(error);
    const errorType = classifyError(error);
    
    // Clear container
    container.innerHTML = '';
    
    // Create error alert
    const alert = document.createElement('div');
    alert.className = `alert alert-danger alert-dismissible fade show`;
    alert.setAttribute('role', 'alert');
    
    alert.innerHTML = `
        <h5 class="alert-heading">
            <i class="bi bi-exclamation-triangle-fill me-2"></i>
            Error
        </h5>
        <p class="mb-2">${escapeHtml(message)}</p>
        ${suggestions.length > 0 ? `
            <hr>
            <p class="mb-1 fw-bold">Try the following:</p>
            <ul class="mb-0">
                ${suggestions.map(s => `<li>${escapeHtml(s)}</li>`).join('')}
            </ul>
        ` : ''}
        ${ERROR_CONFIG.showStackTrace && error.stack ? `
            <hr>
            <details>
                <summary>Technical details</summary>
                <pre class="mt-2 mb-0">${escapeHtml(error.stack)}</pre>
            </details>
        ` : ''}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    container.appendChild(alert);
    container.classList.remove('d-none');
}

/**
 * Clear error from container
 * @param {string} containerId - Container element ID
 */
function clearError(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '';
        container.classList.add('d-none');
    }
}

// ============================================
// ERROR LOGGING
// ============================================

/**
 * Log error to console
 * @param {Error} error - Error object
 */
function logError(error) {
    const errorType = classifyError(error);
    const severity = getErrorSeverity(errorType);
    
    console.group(`[${severity.toUpperCase()}] ${errorType}`);
    console.error('Message:', error.message);
    console.error('Type:', errorType);
    console.error('Severity:', severity);
    if (error.status) console.error('Status:', error.status);
    if (error.code) console.error('Code:', error.code);
    if (error.details) console.error('Details:', error.details);
    if (error.stack) console.error('Stack:', error.stack);
    console.groupEnd();
}

// ============================================
// GLOBAL ERROR HANDLER
// ============================================

/**
 * Setup global error handlers
 */
function setupGlobalErrorHandlers() {
    // Uncaught errors
    window.addEventListener('error', (event) => {
        console.error('Uncaught error:', event.error);
        displayError(event.error || new Error(event.message));
    });
    
    // Unhandled promise rejections
    window.addEventListener('unhandledrejection', (event) => {
        console.error('Unhandled promise rejection:', event.reason);
        displayError(event.reason || new Error('Unhandled promise rejection'));
    });
}

// ============================================
// SUCCESS NOTIFICATIONS
// ============================================

/**
 * Show success message
 * @param {string} message - Success message
 * @param {number} duration - Display duration
 */
function showSuccess(message, duration = ERROR_CONFIG.toastDuration) {
    showToast(message, 'success', duration);
}

/**
 * Show info message
 * @param {string} message - Info message
 * @param {number} duration - Display duration
 */
function showInfo(message, duration = ERROR_CONFIG.toastDuration) {
    showToast(message, 'info', duration);
}

/**
 * Show warning message
 * @param {string} message - Warning message
 * @param {number} duration - Display duration
 */
function showWarning(message, duration = ERROR_CONFIG.toastDuration) {
    showToast(message, 'warning', duration);
}

// ============================================
// CONFIRMATION DIALOGS
// ============================================

/**
 * Show confirmation dialog
 * @param {string} message - Confirmation message
 * @param {Object} options - Dialog options
 * @returns {Promise<boolean>} True if confirmed
 */
async function confirm(message, options = {}) {
    return new Promise((resolve) => {
        // Use native confirm for simplicity (can be enhanced with custom modal)
        const result = window.confirm(message);
        resolve(result);
    });
}

// ============================================
// INITIALIZATION
// ============================================

// Setup global error handlers on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupGlobalErrorHandlers);
} else {
    setupGlobalErrorHandlers();
}

// ============================================
// EXPORTS
// ============================================

if (typeof window !== 'undefined') {
    window.ErrorHandler = {
        // Toast notifications
        showToast,
        showSuccess,
        showInfo,
        showWarning,
        
        // Error display
        displayError,
        showErrorInContainer,
        clearError,
        
        // Error utilities
        classifyError,
        getErrorSeverity,
        getUserFriendlyMessage,
        getRecoverySuggestions,
        
        // Logging
        logError,
        
        // Dialogs
        confirm,
        
        // Constants
        ERROR_TYPES,
        ERROR_SEVERITY,
        
        // Configuration
        config: ERROR_CONFIG
    };
}
