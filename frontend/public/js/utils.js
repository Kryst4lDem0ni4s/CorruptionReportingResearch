/**
 * Corruption Reporting System - DOM Utilities
 * Version: 1.0.0
 * Description: Common DOM manipulation and utility functions
 * 
 * This module provides:
 * - DOM element selection and manipulation
 * - Form validation helpers
 * - URL parameter parsing
 * - Date/time formatting
 * - String utilities
 * - Storage helpers
 * - Debounce/throttle
 * - Animation helpers
 * 
 * Dependencies: None (vanilla JavaScript)
 */

// ============================================
// DOM SELECTION
// ============================================

/**
 * Select single element
 * @param {string} selector - CSS selector
 * @param {Element} context - Context element (default: document)
 * @returns {Element|null} Selected element
 */
function $(selector, context = document) {
    return context.querySelector(selector);
}

/**
 * Select multiple elements
 * @param {string} selector - CSS selector
 * @param {Element} context - Context element (default: document)
 * @returns {NodeList} Selected elements
 */
function $$(selector, context = document) {
    return context.querySelectorAll(selector);
}

/**
 * Get element by ID
 * @param {string} id - Element ID
 * @returns {Element|null} Element
 */
function getById(id) {
    return document.getElementById(id);
}

/**
 * Get elements by class name
 * @param {string} className - Class name
 * @param {Element} context - Context element (default: document)
 * @returns {HTMLCollection} Elements
 */
function getByClass(className, context = document) {
    return context.getElementsByClassName(className);
}

// ============================================
// DOM MANIPULATION
// ============================================

/**
 * Create element with attributes
 * @param {string} tag - HTML tag name
 * @param {Object} attributes - Element attributes
 * @param {string|Element} content - Element content
 * @returns {Element} Created element
 */
function createElement(tag, attributes = {}, content = null) {
    const element = document.createElement(tag);
    
    // Set attributes
    Object.keys(attributes).forEach(key => {
        if (key === 'className') {
            element.className = attributes[key];
        } else if (key === 'dataset') {
            Object.keys(attributes[key]).forEach(dataKey => {
                element.dataset[dataKey] = attributes[key][dataKey];
            });
        } else if (key === 'style' && typeof attributes[key] === 'object') {
            Object.assign(element.style, attributes[key]);
        } else {
            element.setAttribute(key, attributes[key]);
        }
    });
    
    // Set content
    if (content !== null) {
        if (typeof content === 'string') {
            element.textContent = content;
        } else if (content instanceof Element) {
            element.appendChild(content);
        } else if (Array.isArray(content)) {
            content.forEach(child => {
                if (child instanceof Element) {
                    element.appendChild(child);
                }
            });
        }
    }
    
    return element;
}

/**
 * Remove element from DOM
 * @param {Element|string} element - Element or selector
 */
function removeElement(element) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el && el.parentNode) {
        el.parentNode.removeChild(el);
    }
}

/**
 * Empty element (remove all children)
 * @param {Element|string} element - Element or selector
 */
function emptyElement(element) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        while (el.firstChild) {
            el.removeChild(el.firstChild);
        }
    }
}

/**
 * Show element
 * @param {Element|string} element - Element or selector
 * @param {string} display - Display type (default: 'block')
 */
function show(element, display = 'block') {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        el.style.display = display;
        el.classList.remove('d-none');
    }
}

/**
 * Hide element
 * @param {Element|string} element - Element or selector
 */
function hide(element) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        el.style.display = 'none';
        el.classList.add('d-none');
    }
}

/**
 * Toggle element visibility
 * @param {Element|string} element - Element or selector
 */
function toggle(element) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        if (el.style.display === 'none' || el.classList.contains('d-none')) {
            show(el);
        } else {
            hide(el);
        }
    }
}

/**
 * Add class to element
 * @param {Element|string} element - Element or selector
 * @param {string} className - Class name
 */
function addClass(element, className) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        el.classList.add(className);
    }
}

/**
 * Remove class from element
 * @param {Element|string} element - Element or selector
 * @param {string} className - Class name
 */
function removeClass(element, className) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        el.classList.remove(className);
    }
}

/**
 * Toggle class on element
 * @param {Element|string} element - Element or selector
 * @param {string} className - Class name
 */
function toggleClass(element, className) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        el.classList.toggle(className);
    }
}

/**
 * Check if element has class
 * @param {Element|string} element - Element or selector
 * @param {string} className - Class name
 * @returns {boolean} True if has class
 */
function hasClass(element, className) {
    const el = typeof element === 'string' ? $(element) : element;
    return el ? el.classList.contains(className) : false;
}

// ============================================
// URL & QUERY PARAMETERS
// ============================================

/**
 * Get URL query parameter
 * @param {string} name - Parameter name
 * @param {string} url - URL (default: current URL)
 * @returns {string|null} Parameter value
 */
function getQueryParam(name, url = window.location.href) {
    const params = new URLSearchParams(new URL(url).search);
    return params.get(name);
}

/**
 * Get all query parameters
 * @param {string} url - URL (default: current URL)
 * @returns {Object} Parameters object
 */
function getQueryParams(url = window.location.href) {
    const params = new URLSearchParams(new URL(url).search);
    const result = {};
    for (const [key, value] of params) {
        result[key] = value;
    }
    return result;
}

/**
 * Update URL query parameter without reload
 * @param {string} name - Parameter name
 * @param {string} value - Parameter value
 */
function updateQueryParam(name, value) {
    const url = new URL(window.location);
    url.searchParams.set(name, value);
    window.history.pushState({}, '', url);
}

/**
 * Remove query parameter from URL
 * @param {string} name - Parameter name
 */
function removeQueryParam(name) {
    const url = new URL(window.location);
    url.searchParams.delete(name);
    window.history.pushState({}, '', url);
}

// ============================================
// DATE & TIME FORMATTING
// ============================================

/**
 * Format date to readable string
 * @param {Date|string} date - Date object or ISO string
 * @param {Object} options - Intl.DateTimeFormat options
 * @returns {string} Formatted date
 */
function formatDate(date, options = {}) {
    const d = date instanceof Date ? date : new Date(date);
    const defaultOptions = {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    return new Intl.DateTimeFormat('en-US', { ...defaultOptions, ...options }).format(d);
}

/**
 * Get relative time string (e.g., "2 hours ago")
 * @param {Date|string} date - Date object or ISO string
 * @returns {string} Relative time
 */
function getRelativeTime(date) {
    const d = date instanceof Date ? date : new Date(date);
    const now = new Date();
    const diffMs = now - d;
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);
    
    if (diffSecs < 60) return 'just now';
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 30) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    
    return formatDate(d, { year: 'numeric', month: 'short', day: 'numeric' });
}

// ============================================
// STRING UTILITIES
// ============================================

/**
 * Truncate string with ellipsis
 * @param {string} str - String to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} Truncated string
 */
function truncate(str, maxLength) {
    if (!str || str.length <= maxLength) return str;
    return str.substring(0, maxLength - 3) + '...';
}

/**
 * Capitalize first letter
 * @param {string} str - String to capitalize
 * @returns {string} Capitalized string
 */
function capitalize(str) {
    if (!str) return '';
    return str.charAt(0).toUpperCase() + str.slice(1);
}

/**
 * Convert string to title case
 * @param {string} str - String to convert
 * @returns {string} Title case string
 */
function toTitleCase(str) {
    if (!str) return '';
    return str.split(' ')
        .map(word => capitalize(word.toLowerCase()))
        .join(' ');
}

/**
 * Escape HTML special characters
 * @param {string} str - String to escape
 * @returns {string} Escaped string
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

/**
 * Generate random ID
 * @param {number} length - ID length (default: 8)
 * @returns {string} Random ID
 */
function generateId(length = 8) {
    return Math.random().toString(36).substring(2, 2 + length);
}

// ============================================
// VALIDATION
// ============================================

/**
 * Validate email address
 * @param {string} email - Email address
 * @returns {boolean} True if valid
 */
function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
}

/**
 * Validate URL
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
 * Check if string is empty or whitespace
 * @param {string} str - String to check
 * @returns {boolean} True if empty
 */
function isEmpty(str) {
    return !str || str.trim().length === 0;
}

// ============================================
// STORAGE HELPERS
// ============================================

/**
 * Set local storage item
 * @param {string} key - Storage key
 * @param {any} value - Value to store
 */
function setStorage(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
        console.error('Storage error:', error);
    }
}

/**
 * Get local storage item
 * @param {string} key - Storage key
 * @param {any} defaultValue - Default value if not found
 * @returns {any} Stored value
 */
function getStorage(key, defaultValue = null) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : defaultValue;
    } catch (error) {
        console.error('Storage error:', error);
        return defaultValue;
    }
}

/**
 * Remove local storage item
 * @param {string} key - Storage key
 */
function removeStorage(key) {
    try {
        localStorage.removeItem(key);
    } catch (error) {
        console.error('Storage error:', error);
    }
}

/**
 * Clear all local storage
 */
function clearStorage() {
    try {
        localStorage.clear();
    } catch (error) {
        console.error('Storage error:', error);
    }
}

// ============================================
// PERFORMANCE UTILITIES
// ============================================

/**
 * Debounce function execution
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in ms
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle function execution
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in ms
 * @returns {Function} Throttled function
 */
function throttle(func, limit) {
    let inThrottle;
    return function executedFunction(...args) {
        if (!inThrottle) {
            func(...args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Wait for specified time
 * @param {number} ms - Milliseconds to wait
 * @returns {Promise} Promise that resolves after wait
 */
function wait(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// ============================================
// SCROLL UTILITIES
// ============================================

/**
 * Scroll to element smoothly
 * @param {Element|string} element - Element or selector
 * @param {Object} options - Scroll options
 */
function scrollTo(element, options = {}) {
    const el = typeof element === 'string' ? $(element) : element;
    if (el) {
        el.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
            ...options
        });
    }
}

/**
 * Scroll to top of page
 */
function scrollToTop() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

/**
 * Check if element is in viewport
 * @param {Element} element - Element to check
 * @returns {boolean} True if in viewport
 */
function isInViewport(element) {
    const rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}

// ============================================
// FORM UTILITIES
// ============================================

/**
 * Get form data as object
 * @param {HTMLFormElement|string} form - Form element or selector
 * @returns {Object} Form data object
 */
function getFormData(form) {
    const formEl = typeof form === 'string' ? $(form) : form;
    if (!formEl) return {};
    
    const formData = new FormData(formEl);
    const data = {};
    
    for (const [key, value] of formData.entries()) {
        if (data[key]) {
            // Handle multiple values (e.g., checkboxes)
            if (Array.isArray(data[key])) {
                data[key].push(value);
            } else {
                data[key] = [data[key], value];
            }
        } else {
            data[key] = value;
        }
    }
    
    return data;
}

/**
 * Reset form
 * @param {HTMLFormElement|string} form - Form element or selector
 */
function resetForm(form) {
    const formEl = typeof form === 'string' ? $(form) : form;
    if (formEl) {
        formEl.reset();
    }
}

/**
 * Disable form
 * @param {HTMLFormElement|string} form - Form element or selector
 */
function disableForm(form) {
    const formEl = typeof form === 'string' ? $(form) : form;
    if (formEl) {
        const elements = formEl.querySelectorAll('input, select, textarea, button');
        elements.forEach(el => el.disabled = true);
    }
}

/**
 * Enable form
 * @param {HTMLFormElement|string} form - Form element or selector
 */
function enableForm(form) {
    const formEl = typeof form === 'string' ? $(form) : form;
    if (formEl) {
        const elements = formEl.querySelectorAll('input, select, textarea, button');
        elements.forEach(el => el.disabled = false);
    }
}

// ============================================
// COPY TO CLIPBOARD
// ============================================

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} True if successful
 */
async function copyToClipboard(text) {
    try {
        if (navigator.clipboard) {
            await navigator.clipboard.writeText(text);
            return true;
        } else {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            const success = document.execCommand('copy');
            document.body.removeChild(textarea);
            return success;
        }
    } catch (error) {
        console.error('Copy failed:', error);
        return false;
    }
}

// ============================================
// EXPORTS
// ============================================

// Export all utilities
if (typeof window !== 'undefined') {
    window.Utils = {
        // DOM Selection
        $,
        $$,
        getById,
        getByClass,
        
        // DOM Manipulation
        createElement,
        removeElement,
        emptyElement,
        show,
        hide,
        toggle,
        addClass,
        removeClass,
        toggleClass,
        hasClass,
        
        // URL & Query
        getQueryParam,
        getQueryParams,
        updateQueryParam,
        removeQueryParam,
        
        // Date & Time
        formatDate,
        getRelativeTime,
        
        // String Utilities
        truncate,
        capitalize,
        toTitleCase,
        escapeHtml,
        generateId,
        
        // Validation
        isValidEmail,
        isValidUrl,
        isEmpty,
        
        // Storage
        setStorage,
        getStorage,
        removeStorage,
        clearStorage,
        
        // Performance
        debounce,
        throttle,
        wait,
        
        // Scroll
        scrollTo,
        scrollToTop,
        isInViewport,
        
        // Form
        getFormData,
        resetForm,
        disableForm,
        enableForm,
        
        // Clipboard
        copyToClipboard
    };
}
