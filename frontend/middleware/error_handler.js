/**
 * Corruption Reporting System - Express Error Handler Middleware
 * Version: 1.0.0
 * Description: Global error handling middleware for Express
 * 
 * This module provides:
 * - Centralized error handling
 * - Error logging
 * - User-friendly error responses
 * - Stack trace sanitization
 * - Error page rendering
 * 
 * Dependencies: None (uses Express built-in features)
 */

const path = require('path');
const fs = require('fs');

// ============================================
// ERROR TYPES
// ============================================

const ERROR_TYPES = {
    VALIDATION: 'ValidationError',
    NOT_FOUND: 'NotFoundError',
    AUTHENTICATION: 'AuthenticationError',
    AUTHORIZATION: 'AuthorizationError',
    RATE_LIMIT: 'RateLimitError',
    PROXY: 'ProxyError',
    INTERNAL: 'InternalServerError'
};

// ============================================
// ERROR HANDLER CLASS
// ============================================

class ErrorHandler {
    constructor(options = {}) {
        this.options = {
            logErrors: true,
            showStackTrace: process.env.NODE_ENV !== 'production',
            errorLogPath: options.errorLogPath || path.join(__dirname, '../logs/error.log'),
            ...options
        };
        
        // Ensure log directory exists
        this._ensureLogDirectory();
    }
    
    /**
     * Ensure log directory exists
     */
    _ensureLogDirectory() {
        const logDir = path.dirname(this.options.errorLogPath);
        if (!fs.existsSync(logDir)) {
            try {
                fs.mkdirSync(logDir, { recursive: true });
            } catch (error) {
                console.error('Failed to create log directory:', error);
            }
        }
    }
    
    /**
     * Log error to file
     * @param {Error} error - Error object
     * @param {Object} req - Express request
     */
    _logError(error, req) {
        if (!this.options.logErrors) return;
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            method: req.method,
            url: req.url,
            ip: req.ip || req.connection.remoteAddress,
            userAgent: req.get('user-agent'),
            error: {
                name: error.name,
                message: error.message,
                status: error.status || 500,
                stack: error.stack
            }
        };
        
        const logLine = `${timestamp} [ERROR] ${req.method} ${req.url} - ${error.message}\n`;
        
        // Write to console
        console.error(logLine);
        if (this.options.showStackTrace && error.stack) {
            console.error(error.stack);
        }
        
        // Write to file
        try {
            fs.appendFileSync(
                this.options.errorLogPath,
                JSON.stringify(logEntry) + '\n',
                'utf8'
            );
        } catch (err) {
            console.error('Failed to write error log:', err);
        }
    }
    
    /**
     * Get error details
     * @param {Error} error - Error object
     * @returns {Object} Error details
     */
    _getErrorDetails(error) {
        const status = error.status || error.statusCode || 500;
        const message = error.message || 'Internal Server Error';
        const type = error.name || ERROR_TYPES.INTERNAL;
        
        return {
            status,
            message,
            type,
            code: error.code || 'INTERNAL_ERROR',
            details: this.options.showStackTrace ? error.stack : undefined
        };
    }
    
    /**
     * Handle API errors (JSON response)
     * @param {Error} error - Error object
     * @param {Object} req - Express request
     * @param {Object} res - Express response
     */
    _handleAPIError(error, req, res) {
        const { status, message, type, code, details } = this._getErrorDetails(error);
        
        res.status(status).json({
            error: message,
            code,
            type,
            status,
            timestamp: new Date().toISOString(),
            path: req.url,
            details: details ? details : undefined
        });
    }
    
    /**
     * Handle HTML errors (error page)
     * @param {Error} error - Error object
     * @param {Object} req - Express request
     * @param {Object} res - Express response
     */
    _handleHTMLError(error, req, res) {
        const { status, message } = this._getErrorDetails(error);
        
        // Try to send error.html page
        const errorPagePath = path.join(__dirname, '../public/error.html');
        
        if (fs.existsSync(errorPagePath)) {
            // Read and customize error page
            try {
                let html = fs.readFileSync(errorPagePath, 'utf8');
                html = html.replace('{{STATUS}}', status.toString());
                html = html.replace('{{MESSAGE}}', this._escapeHtml(message));
                html = html.replace('{{URL}}', this._escapeHtml(req.url));
                
                res.status(status).send(html);
            } catch (err) {
                // Fallback to simple HTML
                this._sendSimpleErrorHTML(res, status, message);
            }
        } else {
            // Send simple HTML error
            this._sendSimpleErrorHTML(res, status, message);
        }
    }
    
    /**
     * Send simple error HTML
     * @param {Object} res - Express response
     * @param {number} status - HTTP status code
     * @param {string} message - Error message
     */
    _sendSimpleErrorHTML(res, status, message) {
        const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error ${status}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        .error-container {
            background: white;
            border-radius: 10px;
            padding: 40px;
            max-width: 600px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            text-align: center;
        }
        .error-code {
            font-size: 72px;
            font-weight: bold;
            color: #667eea;
            margin: 0;
        }
        .error-message {
            font-size: 24px;
            color: #333;
            margin: 20px 0;
        }
        .error-description {
            color: #666;
            margin: 20px 0;
        }
        .btn {
            display: inline-block;
            padding: 12px 30px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 20px;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #764ba2;
        }
    </style>
</head>
<body>
    <div class="error-container">
        <h1 class="error-code">${status}</h1>
        <h2 class="error-message">${this._escapeHtml(message)}</h2>
        <p class="error-description">
            ${this._getErrorDescription(status)}
        </p>
        <a href="/" class="btn">Go Home</a>
    </div>
</body>
</html>
        `;
        
        res.status(status).send(html);
    }
    
    /**
     * Get user-friendly error description
     * @param {number} status - HTTP status code
     * @returns {string} Error description
     */
    _getErrorDescription(status) {
        const descriptions = {
            400: 'The request could not be understood by the server.',
            401: 'You need to authenticate to access this resource.',
            403: 'You do not have permission to access this resource.',
            404: 'The page you are looking for could not be found.',
            429: 'Too many requests. Please try again later.',
            500: 'The server encountered an unexpected error.',
            502: 'Bad gateway. The upstream server is not responding.',
            503: 'Service temporarily unavailable. Please try again later.'
        };
        
        return descriptions[status] || 'An error occurred while processing your request.';
    }
    
    /**
     * Escape HTML special characters
     * @param {string} text - Text to escape
     * @returns {string} Escaped text
     */
    _escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }
    
    /**
     * Main error handling middleware
     * @param {Error} error - Error object
     * @param {Object} req - Express request
     * @param {Object} res - Express response
     * @param {Function} next - Next middleware
     */
    handle(error, req, res, next) {
        // Log error
        this._logError(error, req);
        
        // Don't handle if headers already sent
        if (res.headersSent) {
            return next(error);
        }
        
        // Determine response type based on Accept header
        const acceptsJSON = req.accepts('json');
        const acceptsHTML = req.accepts('html');
        
        if (acceptsJSON && !acceptsHTML) {
            // API request - send JSON
            this._handleAPIError(error, req, res);
        } else {
            // Browser request - send HTML
            this._handleHTMLError(error, req, res);
        }
    }
}

// ============================================
// 404 NOT FOUND HANDLER
// ============================================

/**
 * 404 Not Found handler
 * @param {Object} req - Express request
 * @param {Object} res - Express response
 * @param {Function} next - Next middleware
 */
function notFoundHandler(req, res, next) {
    const error = new Error(`Not Found: ${req.url}`);
    error.status = 404;
    error.name = ERROR_TYPES.NOT_FOUND;
    next(error);
}

// ============================================
// EXPORTS
// ============================================

// Create default error handler instance
const defaultErrorHandler = new ErrorHandler();

module.exports = {
    ErrorHandler,
    errorHandler: defaultErrorHandler.handle.bind(defaultErrorHandler),
    notFoundHandler,
    ERROR_TYPES
};
