
/**
 * Corruption Reporting System - Evidence Upload Logic
 * Version: 1.0.0
 * Description: Handles evidence submission form, validation, and file uploads
 * 
 * This module manages:
 * - Form validation
 * - File upload handling
 * - Progress tracking
 * - Submission to backend
 * - Success/error handling
 * 
 * Dependencies: api.js, utils.js (optional)
 */

// ============================================
// GLOBAL STATE
// ============================================
let submissionInProgress = false;
let uploadedFiles = {
    images: [],
    audio: [],
    video: [],
    documents: []
};

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    maxFileSize: 50 * 1024 * 1024, // 50MB per file
    maxTotalSize: 100 * 1024 * 1024, // 100MB total
    maxFiles: 20, // Max total files
    allowedExtensions: {
        images: ['jpg', 'jpeg', 'png', 'gif', 'webp'],
        audio: ['mp3', 'wav', 'ogg', 'm4a'],
        video: ['mp4', 'webm', 'mov', 'avi'],
        documents: ['pdf', 'doc', 'docx', 'txt']
    }
};

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize the submission page
 */
function initSubmissionPage() {
    console.log('Initializing submission page...');
    
    // Get form elements
    const form = document.getElementById('submission-form');
    const textInput = document.getElementById('text-input');
    const locationInput = document.getElementById('location-input');
    const submitButton = document.getElementById('submit-button');
    
    // File input elements
    const imageInput = document.getElementById('image-input');
    const audioInput = document.getElementById('audio-input');
    const videoInput = document.getElementById('video-input');
    const documentInput = document.getElementById('document-input');
    
    // Check if elements exist
    if (!form) {
        console.error('Submission form not found');
        return;
    }
    
    // Set up event listeners
    form.addEventListener('submit', handleSubmit);
    
    if (textInput) {
        textInput.addEventListener('input', updateCharCount);
        textInput.addEventListener('input', validateForm);
    }
    
    if (locationInput) {
        locationInput.addEventListener('input', validateForm);
    }
    
    // File input listeners
    if (imageInput) {
        imageInput.addEventListener('change', (e) => handleFileSelect(e, 'images'));
    }
    if (audioInput) {
        audioInput.addEventListener('change', (e) => handleFileSelect(e, 'audio'));
    }
    if (videoInput) {
        videoInput.addEventListener('change', (e) => handleFileSelect(e, 'video'));
    }
    if (documentInput) {
        documentInput.addEventListener('change', (e) => handleFileSelect(e, 'documents'));
    }
    
    // Initial validation
    validateForm();
    
    console.log('Submission page initialized');
}

// ============================================
// VALIDATION
// ============================================

/**
 * Validate the submission form
 * @returns {boolean} True if form is valid
 */
function validateForm() {
    const textInput = document.getElementById('text-input');
    const submitButton = document.getElementById('submit-button');
    
    if (!textInput || !submitButton) return false;
    
    const text = textInput.value.trim();
    const isValid = text.length >= 50; // Minimum 50 characters
    
    // Enable/disable submit button
    submitButton.disabled = !isValid || submissionInProgress;
    
    // Update validation message
    const validationMsg = document.getElementById('text-validation');
    if (validationMsg) {
        if (text.length > 0 && text.length < 50) {
            validationMsg.textContent = `Minimum 50 characters required (${text.length}/50)`;
            validationMsg.className = 'text-danger small mt-1';
        } else if (text.length >= 50) {
            validationMsg.textContent = 'Valid description';
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
    const textInput = document.getElementById('text-input');
    const charCount = document.getElementById('char-count');
    
    if (!textInput || !charCount) return;
    
    const length = textInput.value.length;
    charCount.textContent = `${length} characters`;
    
    // Color coding
    if (length < 50) {
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
 * @param {string} type - File type category
 * @returns {Object} Validation result
 */
function validateFile(file, type) {
    // Check file size
    if (file.size > CONFIG.maxFileSize) {
        return {
            valid: false,
            error: `File "${file.name}" is too large (max ${CONFIG.maxFileSize / 1024 / 1024}MB)`
        };
    }
    
    // Check extension
    const ext = file.name.split('.').pop().toLowerCase();
    if (!CONFIG.allowedExtensions[type].includes(ext)) {
        return {
            valid: false,
            error: `File "${file.name}" has invalid extension. Allowed: ${CONFIG.allowedExtensions[type].join(', ')}`
        };
    }
    
    // Check total file count
    const totalFiles = Object.values(uploadedFiles).reduce((sum, arr) => sum + arr.length, 0);
    if (totalFiles >= CONFIG.maxFiles) {
        return {
            valid: false,
            error: `Maximum ${CONFIG.maxFiles} files allowed`
        };
    }
    
    // Check total size
    const totalSize = Object.values(uploadedFiles)
        .flat()
        .reduce((sum, f) => sum + f.size, 0) + file.size;
    
    if (totalSize > CONFIG.maxTotalSize) {
        return {
            valid: false,
            error: `Total file size exceeds ${CONFIG.maxTotalSize / 1024 / 1024}MB limit`
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
 * @param {string} type - File type category
 */
function handleFileSelect(event, type) {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
        const validation = validateFile(file, type);
        
        if (validation.valid) {
            uploadedFiles[type].push(file);
            addFileToList(file, type);
        } else {
            showError(validation.error);
        }
    });
    
    // Clear input so same file can be selected again
    event.target.value = '';
    
    // Update file count display
    updateFileCount();
}

/**
 * Add file to display list
 * @param {File} file - File to add
 * @param {string} type - File type category
 */
function addFileToList(file, type) {
    const listId = `${type}-list`;
    const list = document.getElementById(listId);
    
    if (!list) return;
    
    const listItem = document.createElement('div');
    listItem.className = 'file-item d-flex justify-content-between align-items-center mb-2 p-2 border rounded';
    listItem.dataset.filename = file.name;
    listItem.dataset.type = type;
    
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
    removeButton.innerHTML = '<i class="bi bi-trash"></i>';
    removeButton.onclick = () => removeFile(file.name, type);
    
    listItem.appendChild(fileInfo);
    listItem.appendChild(removeButton);
    
    list.appendChild(listItem);
}

/**
 * Remove file from upload list
 * @param {string} filename - Filename to remove
 * @param {string} type - File type category
 */
function removeFile(filename, type) {
    // Remove from uploadedFiles
    uploadedFiles[type] = uploadedFiles[type].filter(f => f.name !== filename);
    
    // Remove from DOM
    const listId = `${type}-list`;
    const list = document.getElementById(listId);
    if (list) {
        const items = list.querySelectorAll('.file-item');
        items.forEach(item => {
            if (item.dataset.filename === filename && item.dataset.type === type) {
                item.remove();
            }
        });
    }
    
    // Update file count
    updateFileCount();
}

/**
 * Update file count displays
 */
function updateFileCount() {
    Object.keys(uploadedFiles).forEach(type => {
        const count = uploadedFiles[type].length;
        const badge = document.getElementById(`${type}-count`);
        if (badge) {
            badge.textContent = count;
            badge.className = count > 0 ? 'badge bg-primary ms-2' : 'badge bg-secondary ms-2';
        }
    });
    
    // Update total count
    const totalCount = Object.values(uploadedFiles).reduce((sum, arr) => sum + arr.length, 0);
    const totalBadge = document.getElementById('total-file-count');
    if (totalBadge) {
        totalBadge.textContent = `${totalCount} file(s) selected`;
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
        showError('Please fill in all required fields correctly');
        return;
    }
    
    submissionInProgress = true;
    
    try {
        // Show loading state
        showLoading();
        
        // Build FormData
        const formData = buildFormData();
        
        // Submit to backend
        console.log('Submitting evidence...');
        const result = await api.submitEvidence(formData);
        
        console.log('Submission successful:', result);
        
        // Show success and redirect
        showSuccess(result);
        
    } catch (error) {
        console.error('Submission failed:', error);
        showError(getErrorMessage(error));
    } finally {
        submissionInProgress = false;
        hideLoading();
    }
}

/**
 * Build FormData from form inputs
 * @returns {FormData} Form data
 */
function buildFormData() {
    const formData = new FormData();
    
    // Add text field
    const textInput = document.getElementById('text-input');
    if (textInput) {
        formData.append('text', textInput.value.trim());
    }
    
    // Add location field (optional)
    const locationInput = document.getElementById('location-input');
    if (locationInput && locationInput.value.trim()) {
        formData.append('location', locationInput.value.trim());
    }
    
    // Add files
    Object.keys(uploadedFiles).forEach(type => {
        uploadedFiles[type].forEach(file => {
            formData.append(type, file);
        });
    });
    
    return formData;
}

// ============================================
// UI FEEDBACK
// ============================================

/**
 * Show loading state
 */
function showLoading() {
    const submitButton = document.getElementById('submit-button');
    const loadingOverlay = document.getElementById('loading-overlay');
    const progressSection = document.getElementById('progress-section');
    
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Submitting...';
    }
    
    if (loadingOverlay) {
        loadingOverlay.classList.remove('d-none');
    }
    
    if (progressSection) {
        progressSection.classList.remove('d-none');
        updateProgress(0, 'Preparing submission...');
    }
}

/**
 * Hide loading state
 */
function hideLoading() {
    const submitButton = document.getElementById('submit-button');
    const loadingOverlay = document.getElementById('loading-overlay');
    
    if (submitButton) {
        submitButton.disabled = false;
        submitButton.innerHTML = '<i class="bi bi-upload me-2"></i>Submit Evidence';
    }
    
    if (loadingOverlay) {
        loadingOverlay.classList.add('d-none');
    }
}

/**
 * Update progress bar
 * @param {number} percent - Progress percentage (0-100)
 * @param {string} message - Progress message
 */
function updateProgress(percent, message) {
    const progressBar = document.getElementById('progress-bar');
    const progressText = document.getElementById('progress-text');
    
    if (progressBar) {
        progressBar.style.width = `${percent}%`;
        progressBar.setAttribute('aria-valuenow', percent);
        progressBar.textContent = `${percent}%`;
    }
    
    if (progressText) {
        progressText.textContent = message;
    }
}

/**
 * Show success message and redirect
 * @param {Object} result - Submission result
 */
function showSuccess(result) {
    const successSection = document.getElementById('success-section');
    const submissionIdDisplay = document.getElementById('submission-id-display');
    const pseudonymDisplay = document.getElementById('pseudonym-display');
    const viewResultsLink = document.getElementById('view-results-link');
    
    if (successSection) {
        successSection.classList.remove('d-none');
    }
    
    if (submissionIdDisplay && result.submission_id) {
        submissionIdDisplay.textContent = result.submission_id;
    }
    
    if (pseudonymDisplay && result.pseudonym) {
        pseudonymDisplay.textContent = result.pseudonym;
    }
    
    if (viewResultsLink && result.submission_id) {
        viewResultsLink.href = `/review.html?id=${result.submission_id}`;
    }
    
    // Hide form
    const form = document.getElementById('submission-form');
    if (form) {
        form.classList.add('d-none');
    }
    
    // Auto-redirect after 5 seconds
    setTimeout(() => {
        if (result.submission_id) {
            window.location.href = `/review.html?id=${result.submission_id}`;
        }
    }, 5000);
}

/**
 * Show error message
 * @param {string} message - Error message
 */
function showError(message) {
    const errorSection = document.getElementById('error-section');
    const errorMessage = document.getElementById('error-message');
    
    if (errorSection) {
        errorSection.classList.remove('d-none');
    }
    
    if (errorMessage) {
        errorMessage.textContent = message;
    }
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (errorSection) {
            errorSection.classList.add('d-none');
        }
    }, 5000);
}

/**
 * Reset form
 */
function resetForm() {
    const form = document.getElementById('submission-form');
    if (form) {
        form.reset();
    }
    
    // Clear uploaded files
    uploadedFiles = {
        images: [],
        audio: [],
        video: [],
        documents: []
    };
    
    // Clear file lists
    ['images', 'audio', 'video', 'documents'].forEach(type => {
        const list = document.getElementById(`${type}-list`);
        if (list) {
            list.innerHTML = '';
        }
    });
    
    // Update counts
    updateFileCount();
    updateCharCount();
    validateForm();
    
    // Hide success/error sections
    const successSection = document.getElementById('success-section');
    const errorSection = document.getElementById('error-section');
    if (successSection) successSection.classList.add('d-none');
    if (errorSection) errorSection.classList.add('d-none');
}

// ============================================
// INITIALIZATION ON PAGE LOAD
// ============================================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initSubmissionPage);
} else {
    initSubmissionPage();
}

// Export for external use (if needed)
if (typeof window !== 'undefined') {
    window.SubmissionHandler = {
        init: initSubmissionPage,
        submit: handleSubmit,
        reset: resetForm,
        validateForm,
        uploadedFiles
    };
}
