/**
 * Corruption Reporting System - Client-Side Validators
 * Version: 1.0.0
 * Description: Comprehensive client-side validation utilities
 * 
 * This module provides:
 * - Form field validation
 * - File validation
 * - Input sanitization
 * - Custom validation rules
 * - Error message generation
 * - Real-time validation
 * 
 * Dependencies: None (vanilla JavaScript)
 */

// ============================================
// VALIDATION RULES
// ============================================

const VALIDATION_RULES = {
    text: {
        minLength: 50,
        maxLength: 10000,
        required: true
    },
    counterText: {
        minLength: 100,
        maxLength: 10000,
        required: true
    },
    location: {
        minLength: 0,
        maxLength: 200,
        required: false
    },
    files: {
        maxSize: 50 * 1024 * 1024, // 50MB per file
        maxTotalSize: 100 * 1024 * 1024, // 100MB total
        maxCount: 20,
        allowedExtensions: {
            images: ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp'],
            audio: ['mp3', 'wav', 'ogg', 'm4a', 'aac', 'flac'],
            video: ['mp4', 'webm', 'mov', 'avi', 'mkv'],
            documents: ['pdf', 'doc', 'docx', 'txt', 'rtf', 'odt']
        },
        allowedMimeTypes: {
            images: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'],
            audio: ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp4', 'audio/aac', 'audio/flac'],
            video: ['video/mp4', 'video/webm', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'],
            documents: ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']
        }
    }
};

// ============================================
// TEXT VALIDATION
// ============================================

/**
 * Validate text input
 * @param {string} text - Text to validate
 * @param {Object} rules - Validation rules
 * @returns {Object} Validation result
 */
function validateText(text, rules = VALIDATION_RULES.text) {
    const errors = [];
    
    // Required check
    if (rules.required && (!text || text.trim().length === 0)) {
        errors.push('This field is required');
        return { valid: false, errors };
    }
    
    // If not required and empty, return valid
    if (!rules.required && (!text || text.trim().length === 0)) {
        return { valid: true, errors: [] };
    }
    
    const trimmedText = text.trim();
    
    // Min length
    if (rules.minLength && trimmedText.length < rules.minLength) {
        errors.push(`Minimum ${rules.minLength} characters required (currently ${trimmedText.length})`);
    }
    
    // Max length
    if (rules.maxLength && trimmedText.length > rules.maxLength) {
        errors.push(`Maximum ${rules.maxLength} characters allowed (currently ${trimmedText.length})`);
    }
    
    return {
        valid: errors.length === 0,
        errors,
        length: trimmedText.length
    };
}

/**
 * Validate evidence description
 * @param {string} text - Evidence description
 * @returns {Object} Validation result
 */
function validateEvidenceText(text) {
    return validateText(text, VALIDATION_RULES.text);
}

/**
 * Validate counter-evidence description
 * @param {string} text - Counter-evidence description
 * @returns {Object} Validation result
 */
function validateCounterText(text) {
    return validateText(text, VALIDATION_RULES.counterText);
}

/**
 * Validate location input
 * @param {string} location - Location string
 * @returns {Object} Validation result
 */
function validateLocation(location) {
    return validateText(location, VALIDATION_RULES.location);
}

// ============================================
// FILE VALIDATION
// ============================================

/**
 * Validate file
 * @param {File} file - File to validate
 * @param {string} category - File category (images, audio, video, documents)
 * @returns {Object} Validation result
 */
function validateFile(file, category = 'images') {
    const errors = [];
    const rules = VALIDATION_RULES.files;
    
    if (!file) {
        errors.push('No file provided');
        return { valid: false, errors };
    }
    
    // File size
    if (file.size > rules.maxSize) {
        errors.push(`File "${file.name}" is too large (max ${formatFileSize(rules.maxSize)})`);
    }
    
    // File extension
    const extension = getFileExtension(file.name);
    const allowedExtensions = rules.allowedExtensions[category] || [];
    
    if (!allowedExtensions.includes(extension)) {
        errors.push(`File "${file.name}" has invalid extension. Allowed: ${allowedExtensions.join(', ')}`);
    }
    
    // MIME type (if available)
    if (file.type) {
        const allowedMimeTypes = rules.allowedMimeTypes[category] || [];
        if (allowedMimeTypes.length > 0 && !allowedMimeTypes.includes(file.type)) {
            errors.push(`File "${file.name}" has invalid type: ${file.type}`);
        }
    }
    
    return {
        valid: errors.length === 0,
        errors,
        fileSize: file.size,
        extension,
        mimeType: file.type
    };
}

/**
 * Validate multiple files
 * @param {FileList|Array} files - Files to validate
 * @param {string} category - File category
 * @returns {Object} Validation result
 */
function validateFiles(files, category = 'images') {
    const errors = [];
    const rules = VALIDATION_RULES.files;
    const fileArray = Array.from(files);
    
    // File count
    if (fileArray.length > rules.maxCount) {
        errors.push(`Too many files (max ${rules.maxCount})`);
    }
    
    // Total size
    const totalSize = fileArray.reduce((sum, file) => sum + file.size, 0);
    if (totalSize > rules.maxTotalSize) {
        errors.push(`Total file size too large (max ${formatFileSize(rules.maxTotalSize)})`);
    }
    
    // Validate each file
    const fileResults = fileArray.map(file => validateFile(file, category));
    const invalidFiles = fileResults.filter(result => !result.valid);
    
    // Collect all file errors
    invalidFiles.forEach(result => {
        errors.push(...result.errors);
    });
    
    return {
        valid: errors.length === 0,
        errors,
        fileCount: fileArray.length,
        totalSize,
        fileResults
    };
}

/**
 * Get file extension
 * @param {string} filename - Filename
 * @returns {string} Extension (lowercase, without dot)
 */
function getFileExtension(filename) {
    return filename.split('.').pop().toLowerCase();
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
// FORM VALIDATION
// ============================================

/**
 * Validate submission form
 * @param {Object} formData - Form data object
 * @returns {Object} Validation result
 */
function validateSubmissionForm(formData) {
    const errors = {};
    let valid = true;
    
    // Validate text
    const textResult = validateEvidenceText(formData.text);
    if (!textResult.valid) {
        errors.text = textResult.errors;
        valid = false;
    }
    
    // Validate location (optional)
    if (formData.location) {
        const locationResult = validateLocation(formData.location);
        if (!locationResult.valid) {
            errors.location = locationResult.errors;
            valid = false;
        }
    }
    
    return { valid, errors };
}

/**
 * Validate counter-evidence form
 * @param {Object} formData - Form data object
 * @returns {Object} Validation result
 */
function validateCounterEvidenceForm(formData) {
    const errors = {};
    let valid = true;
    
    // Validate submission ID
    if (!formData.submission_id || formData.submission_id.trim().length === 0) {
        errors.submission_id = ['Submission ID is required'];
        valid = false;
    }
    
    // Validate text
    const textResult = validateCounterText(formData.text);
    if (!textResult.valid) {
        errors.text = textResult.errors;
        valid = false;
    }
    
    return { valid, errors };
}

// ============================================
// INPUT SANITIZATION
// ============================================

/**
 * Sanitize text input
 * @param {string} text - Text to sanitize
 * @returns {string} Sanitized text
 */
function sanitizeText(text) {
    if (!text) return '';
    
    // Trim whitespace
    let sanitized = text.trim();
    
    // Remove null bytes
    sanitized = sanitized.replace(/\0/g, '');
    
    // Normalize whitespace
    sanitized = sanitized.replace(/\s+/g, ' ');
    
    return sanitized;
}

/**
 * Sanitize HTML (basic)
 * @param {string} html - HTML string
 * @returns {string} Sanitized text
 */
function sanitizeHtml(html) {
    const div = document.createElement('div');
    div.textContent = html;
    return div.innerHTML;
}

/**
 * Escape HTML special characters
 * @param {string} text - Text to escape
 * @returns {string} Escaped text
 */
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// ============================================
// REAL-TIME VALIDATION
// ============================================

/**
 * Setup real-time validation for input
 * @param {HTMLElement} input - Input element
 * @param {Function} validator - Validator function
 * @param {HTMLElement} errorContainer - Error display element
 */
function setupRealtimeValidation(input, validator, errorContainer) {
    if (!input) return;
    
    const validate = () => {
        const result = validator(input.value);
        displayValidationResult(input, result, errorContainer);
    };
    
    // Validate on input
    input.addEventListener('input', validate);
    
    // Validate on blur
    input.addEventListener('blur', validate);
    
    // Initial validation
    if (input.value) {
        validate();
    }
}

/**
 * Display validation result
 * @param {HTMLElement} input - Input element
 * @param {Object} result - Validation result
 * @param {HTMLElement} errorContainer - Error display element
 */
function displayValidationResult(input, result, errorContainer) {
    if (!input) return;
    
    // Update input styling
    if (result.valid) {
        input.classList.remove('is-invalid');
        input.classList.add('is-valid');
    } else {
        input.classList.remove('is-valid');
        input.classList.add('is-invalid');
    }
    
    // Display errors
    if (errorContainer) {
        if (result.errors && result.errors.length > 0) {
            errorContainer.innerHTML = result.errors
                .map(error => `<div class="text-danger small">${escapeHtml(error)}</div>`)
                .join('');
            errorContainer.classList.remove('d-none');
        } else {
            errorContainer.innerHTML = '';
            errorContainer.classList.add('d-none');
        }
    }
}

/**
 * Clear validation state
 * @param {HTMLElement} input - Input element
 * @param {HTMLElement} errorContainer - Error display element
 */
function clearValidation(input, errorContainer) {
    if (input) {
        input.classList.remove('is-valid', 'is-invalid');
    }
    
    if (errorContainer) {
        errorContainer.innerHTML = '';
        errorContainer.classList.add('d-none');
    }
}

// ============================================
// VALIDATION HELPERS
// ============================================

/**
 * Check if email is valid
 * @param {string} email - Email address
 * @returns {boolean} True if valid
 */
function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

/**
 * Check if URL is valid
 * @param {string} url - URL string
 * @returns {boolean} True if valid
 */
function isValidUrl(url) {
    try {
        new URL(url);
        return true;
    } catch {
        return false;
    }
}

/**
 * Check if submission ID is valid format
 * @param {string} id - Submission ID
 * @returns {boolean} True if valid
 */
function isValidSubmissionId(id) {
    // Format: sub_[timestamp]_[random]
    const re = /^sub_\d+_[a-z0-9]+$/;
    return re.test(id);
}

// ============================================
// ERROR MESSAGE GENERATION
// ============================================

/**
 * Generate user-friendly error message
 * @param {string} field - Field name
 * @param {Array} errors - Error array
 * @returns {string} Error message
 */
function generateErrorMessage(field, errors) {
    if (!errors || errors.length === 0) return '';
    
    const fieldName = field.charAt(0).toUpperCase() + field.slice(1);
    
    if (errors.length === 1) {
        return `${fieldName}: ${errors[0]}`;
    }
    
    return `${fieldName}:\n${errors.map(e => `- ${e}`).join('\n')}`;
}

// ============================================
// EXPORTS
// ============================================

// Export for CommonJS (Node.js)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        // Rules
        VALIDATION_RULES,
        
        // Text validation
        validateText,
        validateEvidenceText,
        validateCounterText,
        validateLocation,
        
        // File validation
        validateFile,
        validateFiles,
        getFileExtension,
        formatFileSize,
        
        // Form validation
        validateSubmissionForm,
        validateCounterEvidenceForm,
        
        // Sanitization
        sanitizeText,
        sanitizeHtml,
        escapeHtml,
        
        // Real-time validation
        setupRealtimeValidation,
        displayValidationResult,
        clearValidation,
        
        // Helpers
        isValidEmail,
        isValidUrl,
        isValidSubmissionId,
        generateErrorMessage
    };
}

// Export for browser (global)
if (typeof window !== 'undefined') {
    window.Validators = {
        // Rules
        VALIDATION_RULES,
        
        // Text validation
        validateText,
        validateEvidenceText,
        validateCounterText,
        validateLocation,
        
        // File validation
        validateFile,
        validateFiles,
        getFileExtension,
        formatFileSize,
        
        // Form validation
        validateSubmissionForm,
        validateCounterEvidenceForm,
        
        // Sanitization
        sanitizeText,
        sanitizeHtml,
        escapeHtml,
        
        // Real-time validation
        setupRealtimeValidation,
        displayValidationResult,
        clearValidation,
        
        // Helpers
        isValidEmail,
        isValidUrl,
        isValidSubmissionId,
        generateErrorMessage
    };
}
