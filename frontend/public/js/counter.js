/**
 * Corruption Reporting System - Counter-Evidence Logic
 * Version: 1.0.0
 * Description: Handles counter-evidence submission against existing reports
 * 
 * This module manages:
 * - Loading original submission
 * - Counter-evidence form validation
 * - File upload handling
 * - Identity verification checkbox
 * - Submission to backend
 * - Success/error handling
 * 
 * Dependencies: api.js
 */

// ============================================
// GLOBAL STATE
// ============================================
let originalSubmissionId = null;
let originalSubmission = null;
let submissionInProgress = false;
let uploadedEvidenceFiles = [];

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    maxFileSize: 50 * 1024 * 1024, // 50MB per file
    maxFiles: 10, // Max files for counter-evidence
    minTextLength: 100, // Minimum 100 characters for counter-evidence
    allowedExtensions: ['jpg', 'jpeg', 'png', 'pdf', 'doc', 'docx', 'mp4', 'mp3', 'txt']
};

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize the counter-evidence page
 */
function initCounterPage() {
    console.log('Initializing counter-evidence page...');
    
    // Get submission ID from URL
    originalSubmissionId = getSubmissionIdFromURL();
    
    if (!originalSubmissionId) {
        showError('No submission ID provided. Please access this page from a submission review.');
        disableForm();
        return;
    }
    
    // Load original submission
    loadOriginalSubmission(originalSubmissionId);
    
    // Setup form
    setupForm();
    
    console.log('Counter-evidence page initialized');
}

/**
 * Get submission ID from URL parameters
 * @returns {string|null} Submission ID
 */
function getSubmissionIdFromURL() {
    const params = new URLSearchParams(window.location.search);
    return params.get('id');
}

/**
 * Setup form event listeners
 */
function setupForm() {
    const form = document.getElementById('counter-evidence-form');
    const textInput = document.getElementById('counter-text');
    const verifiedCheckbox = document.getElementById('is-verified');
    const fileInput = document.getElementById('evidence-files');
    const submitButton = document.getElementById('submit-counter-btn');
    
    if (!form) {
        console.error('Counter-evidence form not found');
        return;
    }
    
    // Form submission
    form.addEventListener('submit', handleSubmit);
    
    // Text input validation
    if (textInput) {
        textInput.addEventListener('input', updateCharCount);
        textInput.addEventListener('input', validateForm);
    }
    
    // Verified checkbox
    if (verifiedCheckbox) {
        verifiedCheckbox.addEventListener('change', handleVerifiedChange);
    }
    
    // File input
    if (fileInput) {
        fileInput.addEventListener('change', handleFileSelect);
    }
    
    // Initial validation
    validateForm();
}

// ============================================
// DATA LOADING
// ============================================

/**
 * Load original submission data
 * @param {string} submissionId - Submission ID
 */
async function loadOriginalSubmission(submissionId) {
    const loadingSection = document.getElementById('loading-section');
    const errorSection = document.getElementById('error-section');
    const formSection = document.getElementById('form-section');
    
    try {
        if (loadingSection) loadingSection.classList.remove('d-none');
        if (errorSection) errorSection.classList.add('d-none');
        if (formSection) formSection.classList.add('d-none');
        
        console.log(`Loading original submission: ${submissionId}`);
        const submission = await api.getSubmission(submissionId);
        
        originalSubmission = submission;
        
        // Display original submission details
        displayOriginalSubmission(submission);
        
        // Show form
        if (formSection) formSection.classList.remove('d-none');
        
    } catch (error) {
        console.error('Failed to load submission:', error);
        showError(`Failed to load original submission: ${getErrorMessage(error)}`);
        disableForm();
    } finally {
        if (loadingSection) loadingSection.classList.add('d-none');
    }
}

/**
 * Display original submission details
 * @param {Object} submission - Submission data
 */
function displayOriginalSubmission(submission) {
    const originalCard = document.getElementById('original-submission-card');
    if (!originalCard) return;
    
    originalCard.classList.remove('d-none');
    
    // Submission ID
    const idElement = document.getElementById('original-submission-id');
    if (idElement) {
        idElement.textContent = submission.submission_id || 'N/A';
    }
    
    // Timestamp
    const timestampElement = document.getElementById('original-timestamp');
    if (timestampElement) {
        const date = new Date(submission.timestamp);
        timestampElement.textContent = date.toLocaleString();
    }
    
    // Location
    const locationElement = document.getElementById('original-location');
    if (locationElement) {
        locationElement.textContent = submission.location || 'Not specified';
    }
    
    // Credibility score
    const credibilityElement = document.getElementById('original-credibility');
    if (credibilityElement && submission.credibility_assessment) {
        const score = submission.credibility_assessment.overall_score || 0;
        credibilityElement.textContent = (score * 100).toFixed(1) + '%';
        credibilityElement.className = `badge credibility-${getScoreLevel(score)}`;
    }
    
    // Text excerpt (first 200 characters)
    const excerptElement = document.getElementById('original-excerpt');
    if (excerptElement && submission.text) {
        const excerpt = submission.text.length > 200 
            ? submission.text.substring(0, 200) + '...' 
            : submission.text;
        excerptElement.textContent = excerpt;
    }
}

/**
 * Get score level
 * @param {number} score - Score (0-1)
 * @returns {string} Level (low/medium/high)
 */
function getScoreLevel(score) {
    if (score >= 0.75) return 'high';
    if (score >= 0.5) return 'medium';
    return 'low';
}

// ============================================
// VALIDATION
// ============================================

/**
 * Validate the form
 * @returns {boolean} True if form is valid
 */
function validateForm() {
    const textInput = document.getElementById('counter-text');
    const submitButton = document.getElementById('submit-counter-btn');
    
    if (!textInput || !submitButton) return false;
    
    const text = textInput.value.trim();
    const isValid = text.length >= CONFIG.minTextLength;
    
    // Enable/disable submit button
    submitButton.disabled = !isValid || submissionInProgress;
    
    // Update validation message
    const validationMsg = document.getElementById('text-validation');
    if (validationMsg) {
        if (text.length > 0 && text.length < CONFIG.minTextLength) {
            validationMsg.textContent = `Minimum ${CONFIG.minTextLength} characters required (${text.length}/${CONFIG.minTextLength})`;
            validationMsg.className = 'text-danger small mt-1';
        } else if (text.length >= CONFIG.minTextLength) {
            validationMsg.textContent = 'Valid counter-evidence';
            validationMsg.className = 'text-success small mt-1';
        } else {
            validationMsg.textContent = '';
        }
    }
    
    return isValid;
}

/**
 * Update character count display
 */
function updateCharCount() {
    const textInput = document.getElementById('counter-text');
    const charCount = document.getElementById('char-count');
    
    if (!textInput || !charCount) return;
    
    const length = textInput.value.length;
    charCount.textContent = `${length} characters`;
    
    // Color coding
    if (length < CONFIG.minTextLength) {
        charCount.className = 'text-muted small';
    } else if (length < 500) {
        charCount.className = 'text-success small';
    } else {
        charCount.className = 'text-primary small';
    }
}

/**
 * Validate file
 * @param {File} file - File to validate
 * @returns {Object} Validation result
 */
function validateFile(file) {
    // Check file size
    if (file.size > CONFIG.maxFileSize) {
        return {
            valid: false,
            error: `File "${file.name}" is too large (max ${CONFIG.maxFileSize / 1024 / 1024}MB)`
        };
    }
    
    // Check extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!CONFIG.allowedExtensions.includes(ext)) {
        return {
            valid: false,
            error: `File "${file.name}" has invalid extension. Allowed: ${CONFIG.allowedExtensions.join(', ')}`
        };
    }
    
    // Check file count
    if (uploadedEvidenceFiles.length >= CONFIG.maxFiles) {
        return {
            valid: false,
            error: `Maximum ${CONFIG.maxFiles} files allowed`
        };
    }
    
    return { valid: true };
}

// ============================================
// FILE HANDLING
// ============================================

/**
 * Handle file selection
 * @param {Event} event - Change event
 */
function handleFileSelect(event) {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
        const validation = validateFile(file);
        
        if (validation.valid) {
            uploadedEvidenceFiles.push(file);
            addFileToList(file);
        } else {
            showError(validation.error);
        }
    });
    
    // Clear input
    event.target.value = '';
    
    // Update file count
    updateFileCount();
}

/**
 * Add file to display list
 * @param {File} file - File to add
 */
function addFileToList(file) {
    const fileList = document.getElementById('file-list');
    if (!fileList) return;
    
    const listItem = document.createElement('div');
    listItem.className = 'file-item d-flex justify-content-between align-items-center mb-2 p-2 border rounded';
    listItem.dataset.filename = file.name;
    
    const fileInfo = document.createElement('div');
    fileInfo.className = 'file-info';
    
    const fileName = document.createElement('span');
    fileName.className = 'file-name fw-bold';
    fileName.textContent = file.name;
    
    const fileSize = document.createElement('span');
    fileSize.className = 'file-size text-muted small ms-2';
    fileSize.textContent = formatFileSize(file.size);
    
    fileInfo.appendChild(fileName);
    fileInfo.appendChild(fileSize);
    
    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'btn btn-sm btn-danger';
    removeButton.innerHTML = '<i class="bi bi-trash"></i> Remove';
    removeButton.onclick = () => removeFile(file.name);
    
    listItem.appendChild(fileInfo);
    listItem.appendChild(removeButton);
    
    fileList.appendChild(listItem);
}

/**
 * Remove file from upload list
 * @param {string} filename - Filename to remove
 */
function removeFile(filename) {
    // Remove from array
    uploadedEvidenceFiles = uploadedEvidenceFiles.filter(f => f.name !== filename);
    
    // Remove from DOM
    const fileList = document.getElementById('file-list');
    if (fileList) {
        const items = fileList.querySelectorAll('.file-item');
        items.forEach(item => {
            if (item.dataset.filename === filename) {
                item.remove();
            }
        });
    }
    
    // Update count
    updateFileCount();
}

/**
 * Update file count display
 */
function updateFileCount() {
    const badge = document.getElementById('file-count-badge');
    if (badge) {
        const count = uploadedEvidenceFiles.length;
        badge.textContent = count > 0 ? `${count} file(s)` : 'No files';
        badge.className = count > 0 ? 'badge bg-primary' : 'badge bg-secondary';
    }
}

/**
 * Format file size for display
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted size
 */
function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// ============================================
// EVENT HANDLERS
// ============================================

/**
 * Handle verified checkbox change
 */
function handleVerifiedChange() {
    const checkbox = document.getElementById('is-verified');
    const verifiedInfo = document.getElementById('verified-info');
    
    if (!checkbox || !verifiedInfo) return;
    
    if (checkbox.checked) {
        verifiedInfo.classList.remove('d-none');
    } else {
        verifiedInfo.classList.add('d-none');
    }
}

// ============================================
// FORM SUBMISSION
// ============================================

/**
 * Handle form submission
 * @param {Event} event - Submit event
 */
async function handleSubmit(event) {
    event.preventDefault();
    
    if (submissionInProgress) {
        console.warn('Submission already in progress');
        return;
    }
    
    // Validate form
    if (!validateForm()) {
        showError('Please provide a valid counter-evidence description (minimum 100 characters)');
        return;
    }
    
    submissionInProgress = true;
    
    try {
        // Show loading
        showLoading();
        
        // Build submission data
        const data = buildSubmissionData();
        
        // Submit to backend
        console.log('Submitting counter-evidence...');
        const result = await api.submitCounterEvidence(data);
        
        console.log('Counter-evidence submitted successfully:', result);
        
        // Show success
        showSuccess(result);
        
    } catch (error) {
        console.error('Counter-evidence submission failed:', error);
        showError(getErrorMessage(error));
    } finally {
        submissionInProgress = false;
        hideLoading();
    }
}

/**
 * Build submission data
 * @returns {Object} Submission data
 */
function buildSubmissionData() {
    const textInput = document.getElementById('counter-text');
    const verifiedCheckbox = document.getElementById('is-verified');
    
    const data = {
        submission_id: originalSubmissionId,
        text: textInput ? textInput.value.trim() : '',
        is_verified: verifiedCheckbox ? verifiedCheckbox.checked : false,
        evidence_files: uploadedEvidenceFiles
    };
    
    return data;
}

// ============================================
// UI FEEDBACK
// ============================================

/**
 * Show loading state
 */
function showLoading() {
    const submitButton = document.getElementById('submit-counter-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Submitting...';
    }
    
    if (loadingOverlay) {
        loadingOverlay.classList.remove('d-none');
    }
}

/**
 * Hide loading state
 */
function hideLoading() {
    const submitButton = document.getElementById('submit-counter-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (submitButton) {
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="bi bi-send me-2"></i>Submit Counter-Evidence';
    }
    
    if (loadingOverlay) {
        loadingOverlay.classList.add('d-none');
    }
}

/**
 * Show success message
 * @param {Object} result - Submission result
 */
function showSuccess(result) {
    const successSection = document.getElementById('success-section');
    const formSection = document.getElementById('form-section');
    const viewSubmissionLink = document.getElementById('view-submission-link');
    
    if (successSection) {
        successSection.classList.remove('d-none');
    }
    
    if (formSection) {
        formSection.classList.add('d-none');
    }
    
    if (viewSubmissionLink && originalSubmissionId) {
        viewSubmissionLink.href = `/review.html?id=${originalSubmissionId}`;
    }
    
    // Auto-redirect after 5 seconds
    setTimeout(() => {
        if (originalSubmissionId) {
            window.location.href = `/review.html?id=${originalSubmissionId}`;
        }
    }, 5000);
}

/**
 * Show error message
 * @param {string} message - Error message
 */
function showError(message) {
    const errorAlert = document.getElementById('error-alert');
    const errorMessage = document.getElementById('error-message');
    
    if (errorAlert) {
        errorAlert.classList.remove('d-none');
    }
    
    if (errorMessage) {
        errorMessage.textContent = message;
    }
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (errorAlert) {
            errorAlert.classList.add('d-none');
        }
    }, 5000);
}

/**
 * Disable form
 */
function disableForm() {
    const form = document.getElementById('counter-evidence-form');
    if (form) {
        const inputs = form.querySelectorAll('input, textarea, button');
        inputs.forEach(input => {
            input.disabled = true;
        });
    }
}

// ============================================
// INITIALIZATION ON PAGE LOAD
// ============================================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initCounterPage);
} else {
    initCounterPage();
}

// Export for external use
if (typeof window !== 'undefined') {
    window.CounterEvidenceHandler = {
        init: initCounterPage,
        submit: handleSubmit,
        originalSubmission
    };
}
