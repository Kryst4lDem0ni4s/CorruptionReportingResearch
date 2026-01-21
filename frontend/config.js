/**
 * Corruption Reporting System - Frontend Configuration
 * Version: 1.0.0
 * Description: Configuration management for the frontend server
 * 
 * This module provides:
 * - Environment-based configuration
 * - Default values with overrides
 * - Configuration validation
 * - Runtime configuration loading
 * - Support for .env files
 * 
 * Dependencies: None (pure Node.js)
 * 
 * Usage:
 * const config = require('./config');
 * console.log(config.server.port);
 */

const path = require('path');
const fs = require('fs');

// ============================================
// ENVIRONMENT DETECTION
// ============================================

const NODE_ENV = process.env.NODE_ENV || 'development';
const IS_PRODUCTION = NODE_ENV === 'production';
const IS_DEVELOPMENT = NODE_ENV === 'development';
const IS_TEST = NODE_ENV === 'test';

// ============================================
// LOAD ENVIRONMENT VARIABLES
// ============================================

/**
 * Load .env file if it exists
 * This is a simple .env parser to avoid adding dotenv dependency
 */
function loadEnvFile() {
    const envPath = path.join(__dirname, '.env');
    
    if (!fs.existsSync(envPath)) {
        return;
    }
    
    try {
        const envContent = fs.readFileSync(envPath, 'utf8');
        const lines = envContent.split('\n');
        
        lines.forEach(line => {
            // Skip comments and empty lines
            line = line.trim();
            if (!line || line.startsWith('#')) {
                return;
            }
            
            // Parse KEY=VALUE
            const match = line.match(/^([^=]+)=(.*)$/);
            if (match) {
                const key = match[1].trim();
                let value = match[2].trim();
                
                // Remove quotes if present
                if ((value.startsWith('"') && value.endsWith('"')) ||
                    (value.startsWith("'") && value.endsWith("'"))) {
                    value = value.slice(1, -1);
                }
                
                // Only set if not already in environment
                if (!process.env[key]) {
                    process.env[key] = value;
                }
            }
        });
    } catch (error) {
        console.warn('[Config] Warning: Failed to load .env file:', error.message);
    }
}

// Load .env file
loadEnvFile();

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Get environment variable with default value
 * @param {string} key - Environment variable name
 * @param {*} defaultValue - Default value if not set
 * @returns {*} Environment variable value or default
 */
function env(key, defaultValue = undefined) {
    const value = process.env[key];
    return value !== undefined ? value : defaultValue;
}

/**
 * Get integer environment variable
 * @param {string} key - Environment variable name
 * @param {number} defaultValue - Default value
 * @returns {number} Parsed integer value
 */
function envInt(key, defaultValue) {
    const value = process.env[key];
    if (value === undefined) return defaultValue;
    const parsed = parseInt(value, 10);
    return isNaN(parsed) ? defaultValue : parsed;
}

/**
 * Get boolean environment variable
 * @param {string} key - Environment variable name
 * @param {boolean} defaultValue - Default value
 * @returns {boolean} Boolean value
 */
function envBool(key, defaultValue) {
    const value = process.env[key];
    if (value === undefined) return defaultValue;
    return value.toLowerCase() === 'true' || value === '1';
}

// ============================================
// CONFIGURATION OBJECT
// ============================================

const config = {
    // Environment
    env: NODE_ENV,
    isProduction: IS_PRODUCTION,
    isDevelopment: IS_DEVELOPMENT,
    isTest: IS_TEST,
    
    // Server Configuration
    server: {
        port: envInt('PORT', 3000),
        host: env('HOST', '0.0.0.0'),
        env: NODE_ENV,
        name: 'corruption-reporting-frontend',
        version: '1.0.0'
    },
    
    // Backend API Configuration
    backend: {
        url: env('BACKEND_URL', 'http://localhost:8080'),
        apiPrefix: env('BACKEND_API_PREFIX', '/api/v1'),
        timeout: envInt('BACKEND_TIMEOUT', 300000), // 5 minutes
        retries: envInt('BACKEND_RETRIES', 3),
        retryDelay: envInt('BACKEND_RETRY_DELAY', 1000)
    },
    
    // Logging Configuration
    logging: {
        level: env('LOG_LEVEL', IS_PRODUCTION ? 'info' : 'debug'),
        format: env('LOG_FORMAT', IS_PRODUCTION ? 'combined' : 'dev'),
        toFile: envBool('LOG_TO_FILE', IS_PRODUCTION),
        toConsole: envBool('LOG_TO_CONSOLE', true),
        accessLogPath: env('ACCESS_LOG_PATH', path.join(__dirname, 'logs/access.log')),
        errorLogPath: env('ERROR_LOG_PATH', path.join(__dirname, 'logs/error.log')),
        excludePaths: ['/health', '/favicon.ico', '/assets/'],
        colorize: envBool('LOG_COLORIZE', !IS_PRODUCTION)
    },
    
    // Compression Configuration
    compression: {
        enabled: envBool('COMPRESSION_ENABLED', true),
        level: envInt('COMPRESSION_LEVEL', IS_PRODUCTION ? 6 : 1),
        threshold: envInt('COMPRESSION_THRESHOLD', 1024), // 1KB
        memLevel: envInt('COMPRESSION_MEM_LEVEL', 8)
    },
    
    // Security Configuration
    security: {
        // CORS
        cors: {
            enabled: envBool('CORS_ENABLED', true),
            origin: env('CORS_ORIGIN', IS_PRODUCTION ? false : '*'),
            credentials: envBool('CORS_CREDENTIALS', true),
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With'],
            exposedHeaders: ['Content-Length', 'Content-Type'],
            maxAge: envInt('CORS_MAX_AGE', 86400) // 24 hours
        },
        
        // Rate Limiting
        rateLimit: {
            enabled: envBool('RATE_LIMIT_ENABLED', IS_PRODUCTION),
            windowMs: envInt('RATE_LIMIT_WINDOW', 15 * 60 * 1000), // 15 minutes
            max: envInt('RATE_LIMIT_MAX', 100), // requests per window
            message: 'Too many requests, please try again later',
            standardHeaders: true,
            legacyHeaders: false
        },
        
        // Security Headers
        headers: {
            xFrameOptions: env('X_FRAME_OPTIONS', 'DENY'),
            xContentTypeOptions: env('X_CONTENT_TYPE_OPTIONS', 'nosniff'),
            xXssProtection: env('X_XSS_PROTECTION', '1; mode=block'),
            referrerPolicy: env('REFERRER_POLICY', 'strict-origin-when-cross-origin'),
            contentSecurityPolicy: IS_PRODUCTION
        }
    },
    
    // Static Files Configuration
    static: {
        path: env('STATIC_PATH', path.join(__dirname, 'public')),
        maxAge: envInt('STATIC_MAX_AGE', IS_PRODUCTION ? 86400000 : 0), // 1 day in production
        etag: envBool('STATIC_ETAG', true),
        lastModified: envBool('STATIC_LAST_MODIFIED', true),
        index: envBool('STATIC_INDEX', false), // We handle routing manually
        dotfiles: env('STATIC_DOTFILES', 'ignore'),
        extensions: ['html', 'htm']
    },
    
    // Upload Configuration
    upload: {
        maxFileSize: envInt('UPLOAD_MAX_FILE_SIZE', 50 * 1024 * 1024), // 50MB
        maxFiles: envInt('UPLOAD_MAX_FILES', 10),
        allowedTypes: [
            'image/jpeg',
            'image/png',
            'image/gif',
            'image/webp',
            'video/mp4',
            'video/webm',
            'audio/mpeg',
            'audio/wav',
            'application/pdf',
            'text/plain'
        ]
    },
    
    // Proxy Configuration
    proxy: {
        timeout: envInt('PROXY_TIMEOUT', 300000), // 5 minutes
        followRedirects: envBool('PROXY_FOLLOW_REDIRECTS', true),
        maxRedirects: envInt('PROXY_MAX_REDIRECTS', 5),
        validateStatus: (status) => status >= 200 && status < 600
    },
    
    // Session Configuration (for future use)
    session: {
        enabled: envBool('SESSION_ENABLED', false),
        secret: env('SESSION_SECRET', 'change-this-secret-in-production'),
        name: env('SESSION_NAME', 'corruption_reporting_sid'),
        maxAge: envInt('SESSION_MAX_AGE', 24 * 60 * 60 * 1000), // 24 hours
        secure: envBool('SESSION_SECURE', IS_PRODUCTION),
        httpOnly: envBool('SESSION_HTTP_ONLY', true),
        sameSite: env('SESSION_SAME_SITE', 'strict')
    },
    
    // Performance Configuration
    performance: {
        // HTTP Keep-Alive
        keepAliveTimeout: envInt('KEEP_ALIVE_TIMEOUT', 65000),
        headersTimeout: envInt('HEADERS_TIMEOUT', 66000),
        
        // Request Limits
        requestTimeout: envInt('REQUEST_TIMEOUT', 30000),
        bodyLimit: env('BODY_LIMIT', '10mb'),
        
        // Memory Management
        maxOldSpaceSize: envInt('MAX_OLD_SPACE_SIZE', 2048), // 2GB
        
        // Cluster Mode (for future scaling)
        clusterMode: envBool('CLUSTER_MODE', false),
        workers: envInt('WORKERS', require('os').cpus().length)
    },
    
    // Development Configuration
    development: {
        hotReload: envBool('HOT_RELOAD', IS_DEVELOPMENT),
        debugMode: envBool('DEBUG_MODE', IS_DEVELOPMENT),
        verboseLogging: envBool('VERBOSE_LOGGING', IS_DEVELOPMENT),
        mockBackend: envBool('MOCK_BACKEND', false)
    },
    
    // Paths
    paths: {
        root: __dirname,
        public: path.join(__dirname, 'public'),
        logs: path.join(__dirname, 'logs'),
        assets: path.join(__dirname, 'public/assets'),
        uploads: env('UPLOADS_PATH', path.join(__dirname, 'uploads'))
    }
};

// ============================================
// CONFIGURATION VALIDATION
// ============================================

/**
 * Validate configuration
 * @param {Object} cfg - Configuration object
 * @throws {Error} If configuration is invalid
 */
function validateConfig(cfg) {
    const errors = [];
    
    // Validate server port
    if (!cfg.server.port || cfg.server.port < 1 || cfg.server.port > 65535) {
        errors.push(`Invalid port number: ${cfg.server.port}`);
    }
    
    // Validate backend URL
    if (!cfg.backend.url) {
        errors.push('Backend URL is required');
    }
    
    try {
        new URL(cfg.backend.url);
    } catch (e) {
        errors.push(`Invalid backend URL: ${cfg.backend.url}`);
    }
    
    // Validate timeouts
    if (cfg.backend.timeout < 1000) {
        errors.push('Backend timeout must be at least 1000ms');
    }
    
    if (cfg.performance.requestTimeout < 1000) {
        errors.push('Request timeout must be at least 1000ms');
    }
    
    // Validate compression level
    if (cfg.compression.level < 0 || cfg.compression.level > 9) {
        errors.push('Compression level must be between 0 and 9');
    }
    
    // Validate session secret in production
    if (IS_PRODUCTION && cfg.session.enabled && cfg.session.secret === 'change-this-secret-in-production') {
        errors.push('Session secret must be changed in production');
    }
    
    // Throw error if validation failed
    if (errors.length > 0) {
        throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
    }
    
    return true;
}

// Validate configuration
try {
    validateConfig(config);
} catch (error) {
    console.error('[Config] ERROR:', error.message);
    process.exit(1);
}

// ============================================
// CONFIGURATION EXPORT
// ============================================

// Freeze configuration to prevent modification
Object.freeze(config.server);
Object.freeze(config.backend);
Object.freeze(config.logging);
Object.freeze(config.compression);
Object.freeze(config.security);
Object.freeze(config.static);
Object.freeze(config.upload);
Object.freeze(config.proxy);
Object.freeze(config.session);
Object.freeze(config.performance);
Object.freeze(config.development);
Object.freeze(config.paths);
Object.freeze(config);

// Log configuration in development
if (IS_DEVELOPMENT && !IS_TEST) {
    console.log('[Config] Configuration loaded:');
    console.log(`  Environment: ${config.env}`);
    console.log(`  Server: ${config.server.host}:${config.server.port}`);
    console.log(`  Backend: ${config.backend.url}`);
    console.log(`  Logging: ${config.logging.format} (${config.logging.level})`);
    console.log(`  Compression: ${config.compression.enabled ? 'enabled' : 'disabled'} (level ${config.compression.level})`);
}

module.exports = config;
