/**
 * Corruption Reporting System - Request Logger Middleware
 * Version: 1.0.0
 * Description: HTTP request logging middleware
 * 
 * This module provides:
 * - Request logging
 * - Response time tracking
 * - Access log generation
 * - Request/response size tracking
 * 
 * Dependencies: None (uses Node.js built-in modules)
 */

const fs = require('fs');
const path = require('path');

// ============================================
// LOGGER CLASS
// ============================================

class RequestLogger {
    constructor(options = {}) {
        this.options = {
            format: options.format || 'combined', // 'combined', 'common', 'short', 'dev'
            logToConsole: options.logToConsole !== false,
            logToFile: options.logToFile !== false,
            accessLogPath: options.accessLogPath || path.join(__dirname, '../logs/access.log'),
            excludePaths: options.excludePaths || ['/health', '/favicon.ico'],
            colorize: options.colorize !== false && process.env.NODE_ENV !== 'production',
            ...options
        };
        
        // Ensure log directory exists
        this._ensureLogDirectory();
    }
    
    /**
     * Ensure log directory exists
     */
    _ensureLogDirectory() {
        const logDir = path.dirname(this.options.accessLogPath);
        if (!fs.existsSync(logDir)) {
            try {
                fs.mkdirSync(logDir, { recursive: true });
            } catch (error) {
                console.error('Failed to create log directory:', error);
            }
        }
    }
    
    /**
     * Get client IP address
     * @param {Object} req - Express request
     * @returns {string} IP address
     */
    _getIP(req) {
        return req.ip || 
               req.headers['x-forwarded-for'] || 
               req.connection.remoteAddress || 
               'unknown';
    }
    
    /**
     * Get response size
     * @param {Object} res - Express response
     * @returns {number} Response size in bytes
     */
    _getResponseSize(res) {
        const contentLength = res.get('content-length');
        return contentLength ? parseInt(contentLength, 10) : 0;
    }
    
    /**
     * Format log entry
     * @param {Object} req - Express request
     * @param {Object} res - Express response
     * @param {number} responseTime - Response time in ms
     * @returns {string} Formatted log entry
     */
    _formatLog(req, res, responseTime) {
        const timestamp = new Date().toISOString();
        const method = req.method;
        const url = req.originalUrl || req.url;
        const status = res.statusCode;
        const ip = this._getIP(req);
        const userAgent = req.get('user-agent') || '-';
        const responseSize = this._getResponseSize(res);
        const referrer = req.get('referrer') || req.get('referer') || '-';
        
        switch (this.options.format) {
            case 'combined':
                // Apache combined log format
                return `${ip} - - [${timestamp}] "${method} ${url} HTTP/${req.httpVersion}" ${status} ${responseSize} "${referrer}" "${userAgent}"`;
            
            case 'common':
                // Apache common log format
                return `${ip} - - [${timestamp}] "${method} ${url} HTTP/${req.httpVersion}" ${status} ${responseSize}`;
            
            case 'short':
                // Short format
                return `${timestamp} ${method} ${url} ${status} ${responseTime}ms`;
            
            case 'dev':
                // Development format (colorized)
                const statusColor = this._getStatusColor(status);
                const methodColor = '\x1b[35m'; // Magenta
                const resetColor = '\x1b[0m';
                
                if (this.options.colorize) {
                    return `${methodColor}${method}${resetColor} ${url} ${statusColor}${status}${resetColor} ${responseTime}ms`;
                } else {
                    return `${method} ${url} ${status} ${responseTime}ms`;
                }
            
            case 'json':
                // JSON format
                return JSON.stringify({
                    timestamp,
                    method,
                    url,
                    status,
                    responseTime,
                    ip,
                    userAgent,
                    responseSize,
                    referrer
                });
            
            default:
                return `${timestamp} ${method} ${url} ${status} ${responseTime}ms`;
        }
    }
    
    /**
     * Get status code color for terminal
     * @param {number} status - HTTP status code
     * @returns {string} ANSI color code
     */
    _getStatusColor(status) {
        if (status >= 500) return '\x1b[31m'; // Red
        if (status >= 400) return '\x1b[33m'; // Yellow
        if (status >= 300) return '\x1b[36m'; // Cyan
        if (status >= 200) return '\x1b[32m'; // Green
        return '\x1b[0m'; // Reset
    }
    
    /**
     * Write log to file
     * @param {string} logEntry - Log entry
     */
    _writeToFile(logEntry) {
        if (!this.options.logToFile) return;
        
        try {
            fs.appendFileSync(this.options.accessLogPath, logEntry + '\n', 'utf8');
        } catch (error) {
            console.error('Failed to write access log:', error);
        }
    }
    
    /**
     * Write log to console
     * @param {string} logEntry - Log entry
     */
    _writeToConsole(logEntry) {
        if (!this.options.logToConsole) return;
        console.log(logEntry);
    }
    
    /**
     * Check if path should be excluded
     * @param {string} path - Request path
     * @returns {boolean} True if should be excluded
     */
    _shouldExclude(path) {
        return this.options.excludePaths.some(excludePath => {
            if (typeof excludePath === 'string') {
                return path === excludePath;
            } else if (excludePath instanceof RegExp) {
                return excludePath.test(path);
            }
            return false;
        });
    }
    
    /**
     * Logging middleware
     * @param {Object} req - Express request
     * @param {Object} res - Express response
     * @param {Function} next - Next middleware
     */
    log(req, res, next) {
        // Skip excluded paths
        if (this._shouldExclude(req.url)) {
            return next();
        }
        
        const startTime = Date.now();
        
        // Intercept response finish
        const originalEnd = res.end;
        const self = this;
        
        res.end = function(...args) {
            // Calculate response time
            const responseTime = Date.now() - startTime;
            
            // Format and write log
            const logEntry = self._formatLog(req, res, responseTime);
            self._writeToConsole(logEntry);
            self._writeToFile(logEntry);
            
            // Call original end
            originalEnd.apply(res, args);
        };
        
        next();
    }
}

// ============================================
// CONVENIENCE FUNCTIONS
// ============================================

/**
 * Create logger middleware with options
 * @param {Object} options - Logger options
 * @returns {Function} Middleware function
 */
function createLogger(options = {}) {
    const logger = new RequestLogger(options);
    return logger.log.bind(logger);
}

/**
 * Development logger (colorized, console only)
 * @returns {Function} Middleware function
 */
function devLogger() {
    return createLogger({
        format: 'dev',
        logToFile: false,
        logToConsole: true,
        colorize: true
    });
}

/**
 * Production logger (combined format, file + console)
 * @returns {Function} Middleware function
 */
function prodLogger() {
    return createLogger({
        format: 'combined',
        logToFile: true,
        logToConsole: true,
        colorize: false
    });
}

// ============================================
// EXPORTS
// ============================================

// Create default logger
const defaultLogger = new RequestLogger();

module.exports = {
    RequestLogger,
    logger: defaultLogger.log.bind(defaultLogger),
    createLogger,
    devLogger,
    prodLogger
};
