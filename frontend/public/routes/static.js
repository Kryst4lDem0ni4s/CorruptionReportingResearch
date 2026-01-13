/**
 * Corruption Reporting System - Static File Handler
 * Version: 1.0.0
 * Description: Serves static files (HTML, CSS, JS, images) with caching and security
 * 
 * This router handles serving all static assets from the public directory,
 * including proper MIME types, caching headers, security headers, and ETag support.
 */

const express = require('express');
const path = require('path');
const fs = require('fs');

const router = express.Router();

// ============================================
// CONFIGURATION
// ============================================
const PUBLIC_DIR = path.join(__dirname, '..', 'public');
const ENABLE_CACHE = process.env.NODE_ENV === 'production';
const CACHE_MAX_AGE = parseInt(process.env.CACHE_MAX_AGE || '86400', 10); // 1 day in seconds

// ============================================
// SECURITY HEADERS
// ============================================

/**
 * Set security headers for static files
 * @param {Object} res - Express response object
 * @param {string} filePath - Path to the file being served
 */
function setSecurityHeaders(res, filePath) {
    // Prevent MIME type sniffing
    res.setHeader('X-Content-Type-Options', 'nosniff');
    
    // Prevent clickjacking
    res.setHeader('X-Frame-Options', 'DENY');
    
    // XSS protection (legacy but still useful)
    res.setHeader('X-XSS-Protection', '1; mode=block');
    
    // Content Security Policy (basic)
    if (filePath.endsWith('.html')) {
        res.setHeader(
            'Content-Security-Policy',
            "default-src 'self'; " +
            "script-src 'self' 'unsafe-inline'; " +
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; " +
            "font-src 'self' https://cdn.jsdelivr.net; " +
            "img-src 'self' data: https:; " +
            "connect-src 'self';"
        );
    }
    
    // Referrer policy
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
}

/**
 * Set cache headers for static files
 * @param {Object} res - Express response object
 * @param {string} filePath - Path to the file being served
 */
function setCacheHeaders(res, filePath) {
    if (!ENABLE_CACHE) {
        // Disable caching in development
        res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
        res.setHeader('Pragma', 'no-cache');
        res.setHeader('Expires', '0');
        return;
    }
    
    // Different cache strategies based on file type
    const ext = path.extname(filePath).toLowerCase();
    
    // HTML files: short cache (can change frequently)
    if (ext === '.html') {
        res.setHeader('Cache-Control', 'public, max-age=3600'); // 1 hour
    }
    // CSS and JS: medium cache with validation
    else if (ext === '.css' || ext === '.js') {
        res.setHeader('Cache-Control', `public, max-age=${CACHE_MAX_AGE}, must-revalidate`);
    }
    // Images and other assets: long cache
    else if (['.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.woff', '.woff2', '.ttf'].includes(ext)) {
        res.setHeader('Cache-Control', `public, max-age=${CACHE_MAX_AGE * 7}, immutable`); // 7 days
    }
    // Default: standard cache
    else {
        res.setHeader('Cache-Control', `public, max-age=${CACHE_MAX_AGE}`);
    }
}

/**
 * Get MIME type for file
 * @param {string} filePath - Path to the file
 * @returns {string} MIME type
 */
function getMimeType(filePath) {
    const ext = path.extname(filePath).toLowerCase();
    
    const mimeTypes = {
        '.html': 'text/html; charset=utf-8',
        '.css': 'text/css; charset=utf-8',
        '.js': 'application/javascript; charset=utf-8',
        '.json': 'application/json; charset=utf-8',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain; charset=utf-8',
        '.xml': 'application/xml; charset=utf-8'
    };
    
    return mimeTypes[ext] || 'application/octet-stream';
}

/**
 * Validate file path to prevent directory traversal
 * @param {string} filePath - Requested file path
 * @returns {boolean} True if path is safe
 */
function isPathSafe(filePath) {
    const normalizedPath = path.normalize(filePath);
    const resolvedPath = path.resolve(PUBLIC_DIR, normalizedPath);
    
    // Ensure the resolved path is within the public directory
    return resolvedPath.startsWith(PUBLIC_DIR);
}

// ============================================
// STATIC FILE MIDDLEWARE
// ============================================

// Express static middleware with custom options
router.use(express.static(PUBLIC_DIR, {
    dotfiles: 'deny', // Don't serve dotfiles
    etag: true, // Enable ETag generation
    extensions: false, // Don't automatically add extensions
    index: false, // Don't serve index.html automatically (handled by routes)
    maxAge: ENABLE_CACHE ? CACHE_MAX_AGE * 1000 : 0, // maxAge in milliseconds
    redirect: false, // Don't redirect to trailing slash
    setHeaders: (res, filePath) => {
        setSecurityHeaders(res, filePath);
        setCacheHeaders(res, filePath);
        
        // Set MIME type
        const mimeType = getMimeType(filePath);
        res.setHeader('Content-Type', mimeType);
        
        // Log in development
        if (process.env.NODE_ENV === 'development') {
            const relativePath = path.relative(PUBLIC_DIR, filePath);
            console.log(`Serving static file: ${relativePath}`);
        }
    }
}));

// ============================================
// CUSTOM FILE HANDLER (FOR EXPLICIT CONTROL)
// ============================================

/**
 * Serve a specific file with custom handling
 * @param {string} requestedPath - Requested file path
 * @param {Object} req - Express request
 * @param {Object} res - Express response
 * @param {Function} next - Express next function
 */
function serveFile(requestedPath, req, res, next) {
    // Security check
    if (!isPathSafe(requestedPath)) {
        console.warn(`Blocked unsafe path access: ${requestedPath}`);
        return res.status(403).json({
            error: 'Forbidden',
            message: 'Access to this path is not allowed'
        });
    }
    
    const filePath = path.join(PUBLIC_DIR, requestedPath);
    
    // Check if file exists
    fs.access(filePath, fs.constants.R_OK, (err) => {
        if (err) {
            return next(); // File not found, pass to next middleware
        }
        
        // Get file stats for ETag and size
        fs.stat(filePath, (err, stats) => {
            if (err) {
                return next(err);
            }
            
            // Check if it's a file (not a directory)
            if (!stats.isFile()) {
                return next();
            }
            
            // Set headers
            setSecurityHeaders(res, filePath);
            setCacheHeaders(res, filePath);
            
            const mimeType = getMimeType(filePath);
            res.setHeader('Content-Type', mimeType);
            res.setHeader('Content-Length', stats.size);
            
            // Generate simple ETag based on mtime and size
            const etag = `"${stats.size}-${stats.mtime.getTime()}"`;
            res.setHeader('ETag', etag);
            
            // Check If-None-Match header for 304 Not Modified
            const ifNoneMatch = req.headers['if-none-match'];
            if (ifNoneMatch === etag) {
                return res.status(304).end();
            }
            
            // Stream the file
            const readStream = fs.createReadStream(filePath);
            
            readStream.on('error', (error) => {
                console.error(`Error streaming file ${filePath}:`, error);
                if (!res.headersSent) {
                    return res.status(500).json({
                        error: 'Internal server error',
                        message: 'Error reading file'
                    });
                }
            });
            
            readStream.pipe(res);
        });
    });
}

// ============================================
// CUSTOM ROUTES (IF NEEDED)
// ============================================

// Explicitly handle CSS files
router.get('/css/:filename', (req, res, next) => {
    const filename = req.params.filename;
    serveFile(`css/${filename}`, req, res, next);
});

// Explicitly handle JS files
router.get('/js/:filename', (req, res, next) => {
    const filename = req.params.filename;
    serveFile(`js/${filename}`, req, res, next);
});

// Explicitly handle assets
router.get('/assets/:filename', (req, res, next) => {
    const filename = req.params.filename;
    serveFile(`assets/${filename}`, req, res, next);
});

// ============================================
// ERROR HANDLING
// ============================================

// 404 handler for static files (if express.static didn't catch it)
router.use((req, res, next) => {
    // This will be caught by the global 404 handler in index.js
    next();
});

// ============================================
// UTILITY FUNCTIONS
// ============================================

/**
 * Get file info for debugging
 * @param {string} filename - Name of the file
 * @returns {Object} File information
 */
function getFileInfo(filename) {
    const filePath = path.join(PUBLIC_DIR, filename);
    
    try {
        const stats = fs.statSync(filePath);
        return {
            exists: true,
            size: stats.size,
            mtime: stats.mtime,
            isFile: stats.isFile(),
            isDirectory: stats.isDirectory()
        };
    } catch (error) {
        return {
            exists: false,
            error: error.message
        };
    }
}

/**
 * List all files in public directory (for debugging)
 * @returns {Array} List of files
 */
function listPublicFiles() {
    const files = [];
    
    function walkDir(dir, prefix = '') {
        try {
            const items = fs.readdirSync(dir);
            
            items.forEach(item => {
                const fullPath = path.join(dir, item);
                const relativePath = path.join(prefix, item);
                const stats = fs.statSync(fullPath);
                
                if (stats.isDirectory()) {
                    walkDir(fullPath, relativePath);
                } else {
                    files.push({
                        path: relativePath,
                        size: stats.size,
                        mtime: stats.mtime
                    });
                }
            });
        } catch (error) {
            console.error(`Error reading directory ${dir}:`, error);
        }
    }
    
    walkDir(PUBLIC_DIR);
    return files;
}

// ============================================
// EXPORTS
// ============================================
module.exports = router;
module.exports.getFileInfo = getFileInfo;
module.exports.listPublicFiles = listPublicFiles;
