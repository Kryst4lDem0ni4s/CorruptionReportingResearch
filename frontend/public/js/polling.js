/**
 * Corruption Reporting System - Status Polling
 * Version: 1.0.0
 * Description: Automatic status polling system for long-running operations
 * 
 * This module provides:
 * - Automatic submission status polling
 * - Configurable polling intervals
 * - Exponential backoff
 * - Event-driven updates
 * - Automatic stop on completion
 * - Error handling and recovery
 * 
 * Dependencies: api.js
 */

// ============================================
// CONFIGURATION
// ============================================
const POLLING_CONFIG = {
    defaultInterval: 2000, // 2 seconds
    maxInterval: 10000, // 10 seconds (exponential backoff limit)
    maxAttempts: 150, // 5 minutes at 2s intervals
    backoffMultiplier: 1.5, // Exponential backoff multiplier
    enableBackoff: true,
    enableEventEmitter: true
};

// ============================================
// POLLING MANAGER
// ============================================

class PollingManager {
    constructor(config = {}) {
        this.config = { ...POLLING_CONFIG, ...config };
        this.activePollers = new Map();
        this.pollerId = 0;
        
        // Event listeners
        this.listeners = {
            update: [],
            complete: [],
            error: [],
            timeout: []
        };
    }
    
    /**
     * Start polling for submission status
     * @param {string} submissionId - Submission ID
     * @param {Object} options - Polling options
     * @returns {number} Poller ID
     */
    startPolling(submissionId, options = {}) {
        const pollerId = ++this.pollerId;
        
        const poller = {
            id: pollerId,
            submissionId,
            interval: options.interval || this.config.defaultInterval,
            maxAttempts: options.maxAttempts || this.config.maxAttempts,
            attempts: 0,
            currentInterval: options.interval || this.config.defaultInterval,
            timeoutId: null,
            active: true,
            startTime: Date.now(),
            onUpdate: options.onUpdate || null,
            onComplete: options.onComplete || null,
            onError: options.onError || null,
            onTimeout: options.onTimeout || null
        };
        
        this.activePollers.set(pollerId, poller);
        this._poll(pollerId);
        
        return pollerId;
    }
    
    /**
     * Stop polling
     * @param {number} pollerId - Poller ID
     */
    stopPolling(pollerId) {
        const poller = this.activePollers.get(pollerId);
        if (!poller) return;
        
        poller.active = false;
        
        if (poller.timeoutId) {
            clearTimeout(poller.timeoutId);
            poller.timeoutId = null;
        }
        
        this.activePollers.delete(pollerId);
    }
    
    /**
     * Stop all active pollers
     */
    stopAll() {
        this.activePollers.forEach((poller, pollerId) => {
            this.stopPolling(pollerId);
        });
    }
    
    /**
     * Get active poller count
     * @returns {number} Active poller count
     */
    getActiveCount() {
        return this.activePollers.size;
    }
    
    /**
     * Check if polling for submission
     * @param {string} submissionId - Submission ID
     * @returns {boolean} True if polling
     */
    isPolling(submissionId) {
        for (const poller of this.activePollers.values()) {
            if (poller.submissionId === submissionId && poller.active) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * Internal polling function
     * @param {number} pollerId - Poller ID
     */
    async _poll(pollerId) {
        const poller = this.activePollers.get(pollerId);
        if (!poller || !poller.active) return;
        
        try {
            // Fetch submission status
            const submission = await api.getSubmission(poller.submissionId);
            
            poller.attempts++;
            
            // Emit update event
            this._emitUpdate(poller, submission);
            
            // Check if complete
            if (submission.status === 'completed') {
                this._emitComplete(poller, submission);
                this.stopPolling(pollerId);
                return;
            }
            
            // Check if failed
            if (submission.status === 'failed') {
                this._emitError(poller, new Error(submission.error_message || 'Processing failed'), submission);
                this.stopPolling(pollerId);
                return;
            }
            
            // Check max attempts
            if (poller.attempts >= poller.maxAttempts) {
                this._emitTimeout(poller);
                this.stopPolling(pollerId);
                return;
            }
            
            // Schedule next poll with exponential backoff
            if (this.config.enableBackoff) {
                poller.currentInterval = Math.min(
                    poller.currentInterval * this.config.backoffMultiplier,
                    this.config.maxInterval
                );
            }
            
            poller.timeoutId = setTimeout(() => this._poll(pollerId), poller.currentInterval);
            
        } catch (error) {
            console.error('Polling error:', error);
            
            // Emit error event
            this._emitError(poller, error);
            
            // Retry if not max attempts
            if (poller.attempts < poller.maxAttempts) {
                poller.attempts++;
                poller.timeoutId = setTimeout(() => this._poll(pollerId), poller.currentInterval);
            } else {
                this._emitTimeout(poller);
                this.stopPolling(pollerId);
            }
        }
    }
    
    /**
     * Emit update event
     * @param {Object} poller - Poller object
     * @param {Object} submission - Submission data
     */
    _emitUpdate(poller, submission) {
        // Call individual callback
        if (poller.onUpdate) {
            poller.onUpdate(submission);
        }
        
        // Call global listeners
        this.listeners.update.forEach(listener => {
            try {
                listener(submission, poller);
            } catch (error) {
                console.error('Update listener error:', error);
            }
        });
    }
    
    /**
     * Emit complete event
     * @param {Object} poller - Poller object
     * @param {Object} submission - Submission data
     */
    _emitComplete(poller, submission) {
        // Call individual callback
        if (poller.onComplete) {
            poller.onComplete(submission);
        }
        
        // Call global listeners
        this.listeners.complete.forEach(listener => {
            try {
                listener(submission, poller);
            } catch (error) {
                console.error('Complete listener error:', error);
            }
        });
    }
    
    /**
     * Emit error event
     * @param {Object} poller - Poller object
     * @param {Error} error - Error object
     * @param {Object} submission - Submission data (optional)
     */
    _emitError(poller, error, submission = null) {
        // Call individual callback
        if (poller.onError) {
            poller.onError(error, submission);
        }
        
        // Call global listeners
        this.listeners.error.forEach(listener => {
            try {
                listener(error, poller, submission);
            } catch (err) {
                console.error('Error listener error:', err);
            }
        });
    }
    
    /**
     * Emit timeout event
     * @param {Object} poller - Poller object
     */
    _emitTimeout(poller) {
        const elapsed = Date.now() - poller.startTime;
        
        // Call individual callback
        if (poller.onTimeout) {
            poller.onTimeout(elapsed, poller.attempts);
        }
        
        // Call global listeners
        this.listeners.timeout.forEach(listener => {
            try {
                listener(poller, elapsed);
            } catch (error) {
                console.error('Timeout listener error:', error);
            }
        });
    }
    
    /**
     * Add event listener
     * @param {string} event - Event name (update, complete, error, timeout)
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (this.listeners[event]) {
            this.listeners[event].push(callback);
        }
    }
    
    /**
     * Remove event listener
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    off(event, callback) {
        if (this.listeners[event]) {
            const index = this.listeners[event].indexOf(callback);
            if (index > -1) {
                this.listeners[event].splice(index, 1);
            }
        }
    }
    
    /**
     * Get poller info
     * @param {number} pollerId - Poller ID
     * @returns {Object} Poller info
     */
    getPollerInfo(pollerId) {
        const poller = this.activePollers.get(pollerId);
        if (!poller) return null;
        
        return {
            id: poller.id,
            submissionId: poller.submissionId,
            attempts: poller.attempts,
            maxAttempts: poller.maxAttempts,
            currentInterval: poller.currentInterval,
            active: poller.active,
            elapsed: Date.now() - poller.startTime
        };
    }
}

// ============================================
// CONVENIENCE FUNCTIONS
// ============================================

/**
 * Simple polling function
 * @param {string} submissionId - Submission ID
 * @param {Function} onUpdate - Update callback
 * @param {Object} options - Polling options
 * @returns {Promise<Object>} Final submission data
 */
function pollSubmission(submissionId, onUpdate, options = {}) {
    return new Promise((resolve, reject) => {
        const manager = getPollingManager();
        
        manager.startPolling(submissionId, {
            ...options,
            onUpdate,
            onComplete: (submission) => resolve(submission),
            onError: (error) => reject(error),
            onTimeout: () => reject(new Error('Polling timed out'))
        });
    });
}

/**
 * Poll with progress tracking
 * @param {string} submissionId - Submission ID
 * @param {Object} progressElements - DOM elements for progress display
 * @param {Object} options - Polling options
 * @returns {Promise<Object>} Final submission data
 */
function pollWithProgress(submissionId, progressElements = {}, options = {}) {
    const {
        progressBar,
        progressText,
        statusText,
        stageText
    } = progressElements;
    
    return pollSubmission(
        submissionId,
        (submission) => {
            // Update progress bar
            if (progressBar) {
                const progress = getProgressFromStatus(submission.status, submission.processing_stage);
                progressBar.style.width = `${progress}%`;
                progressBar.setAttribute('aria-valuenow', progress);
            }
            
            // Update progress text
            if (progressText) {
                progressText.textContent = getProgressMessage(submission.processing_stage);
            }
            
            // Update status
            if (statusText) {
                statusText.textContent = submission.status;
            }
            
            // Update stage
            if (stageText) {
                stageText.textContent = submission.processing_stage || 'unknown';
            }
        },
        options
    );
}

/**
 * Get progress percentage from status
 * @param {string} status - Status
 * @param {string} stage - Processing stage
 * @returns {number} Progress percentage
 */
function getProgressFromStatus(status, stage) {
    if (status === 'completed') return 100;
    if (status === 'failed') return 0;
    
    const stages = {
        'anonymization': 20,
        'credibility_assessment': 40,
        'coordination_detection': 60,
        'consensus': 80,
        'reporting': 90
    };
    
    return stages[stage] || 10;
}

/**
 * Get progress message
 * @param {string} stage - Processing stage
 * @returns {string} Progress message
 */
function getProgressMessage(stage) {
    const messages = {
        'anonymization': 'Anonymizing submission...',
        'credibility_assessment': 'Analyzing evidence credibility...',
        'coordination_detection': 'Detecting coordination patterns...',
        'consensus': 'Building consensus...',
        'reporting': 'Generating report...'
    };
    
    return messages[stage] || 'Processing...';
}

// ============================================
// SINGLETON INSTANCE
// ============================================

let defaultPollingManager = null;

/**
 * Get or create default polling manager
 * @returns {PollingManager} Polling manager instance
 */
function getPollingManager() {
    if (!defaultPollingManager) {
        defaultPollingManager = new PollingManager();
    }
    return defaultPollingManager;
}

// ============================================
// AUTO-CLEANUP
// ============================================

// Stop all polling on page unload
if (typeof window !== 'undefined') {
    window.addEventListener('beforeunload', () => {
        const manager = getPollingManager();
        manager.stopAll();
    });
}

// ============================================
// EXPORTS
// ============================================

if (typeof window !== 'undefined') {
    window.PollingManager = PollingManager;
    window.pollingManager = getPollingManager();
    window.pollSubmission = pollSubmission;
    window.pollWithProgress = pollWithProgress;
}
