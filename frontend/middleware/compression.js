/**
 * Corruption Reporting System - Compression Middleware
 * Version: 1.0.0
 * Description: Response compression middleware
 * 
 * This module provides:
 * - Gzip/Deflate compression
 * - Configurable compression levels
 * - File type filtering
 * - Size threshold configuration
 * 
 * Dependencies: compression (npm package - already in package.json)
 */

const compression = require('compression');

// ============================================
// COMPRESSION CONFIGURATION
// ============================================

const COMPRESSION_CONFIG = {
    // Compression level (0-9, 9 = max compression)
    level: 6,
    
    // Minimum size to compress (bytes)
    threshold: 1024, // 1KB
    
    // File types to compress
    compressibleTypes: [
        'text/html',
        'text/css',
        'text/javascript',
        'text/plain',
        'text/xml',
        'application/json',
        'application/javascript',
        'application/xml',
        'application/xml+rss',
        'application/xhtml+xml',
        'application/x-font-ttf',
        'application/x-font-opentype',
        'application/vnd.ms-fontobject',
        'image/svg+xml',
        'image/x-icon'
    ],
    
    // File extensions to never compress (already compressed)
    excludeExtensions: [
        '.jpg', '.jpeg', '.png', '.gif', '.webp',
        '.mp3', '.mp4', '.avi', '.mov',
        '.zip', '.rar', '.7z', '.gz',
        '.pdf'
    ]
};

// ============================================
// COMPRESSION MIDDLEWARE
// ============================================

/**
 * Filter function to determine if response should be compressed
 * @param {Object} req - Express request
 * @param {Object} res - Express response
 * @returns {boolean} True if should compress
 */
function shouldCompress(req, res) {
    // Don't compress if client doesn't accept encoding
    if (!req.headers['accept-encoding']) {
        return false;
    }
    
    // Check for x-no-compression header
    if (req.headers['x-no-compression']) {
        return false;
    }
    
    // Check file extension
    const url = req.url.split('?')[0];
    const hasExcludedExtension = COMPRESSION_CONFIG.excludeExtensions.some(ext => {
        return url.toLowerCase().endsWith(ext);
    });
    
    if (hasExcludedExtension) {
        return false;
    }
    
    // Use default compression filter
    return compression.filter(req, res);
}

/**
 * Create compression middleware
 * @param {Object} options - Compression options
 * @returns {Function} Middleware function
 */
function createCompression(options = {}) {
    const config = {
        ...COMPRESSION_CONFIG,
        ...options
    };
    
    return compression({
        // Compression level
        level: config.level,
        
        // Minimum threshold
        threshold: config.threshold,
        
        // Filter function
        filter: options.filter || shouldCompress,
        
        // Memory level (1-9)
        memLevel: 8,
        
        // Window bits
        windowBits: 15,
        
        // Strategy
        strategy: require('zlib').Z_DEFAULT_STRATEGY
    });
}

/**
 * Development compression (lower level for speed)
 * @returns {Function} Middleware function
 */
function devCompression() {
    return createCompression({
        level: 1, // Fastest compression
        threshold: 2048 // 2KB
    });
}

/**
 * Production compression (higher level for size)
 * @returns {Function} Middleware function
 */
function prodCompression() {
    return createCompression({
        level: 6, // Balanced compression
        threshold: 1024 // 1KB
    });
}

/**
 * Maximum compression (slowest but smallest)
 * @returns {Function} Middleware function
 */
function maxCompression() {
    return createCompression({
        level: 9, // Maximum compression
        threshold: 512 // 512 bytes
    });
}

// ============================================
// EXPORTS
// ============================================

module.exports = {
    compression: createCompression(),
    createCompression,
    devCompression,
    prodCompression,
    maxCompression,
    shouldCompress,
    COMPRESSION_CONFIG
};
