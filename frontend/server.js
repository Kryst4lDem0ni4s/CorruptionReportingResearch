/**
 * Corruption Reporting System - Frontend Server
 * Version: 1.0.0
 * Description: Express server entry point for the corruption reporting frontend
 * 
 * This server provides:
 * - Static file serving for HTML/CSS/JS
 * - API proxy to Python backend
 * - Request logging and compression
 * - Error handling
 * - Graceful shutdown
 * 
 * Dependencies: express, axios, compression
 * 
 * Usage:
 * npm start                    # Production mode
 * npm run dev                  # Development mode
 * NODE_ENV=production node server.js
 */

const express = require('express');
const path = require('path');
const fs = require('fs');

// ============================================
// ENVIRONMENT & CONFIGURATION
// ============================================

const NODE_ENV = process.env.NODE_ENV || 'development';
const PORT = parseInt(process.env.PORT || '3000', 10);
const HOST = process.env.HOST || '0.0.0.0';
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Import configuration if available
let config = {};
try {
    const configPath = path.join(__dirname, 'config.js');
    if (fs.existsSync(configPath)) {
        config = require('./config');
    }
} catch (error) {
    console.warn('Config file not found, using environment variables');
}

// ============================================
// IMPORT MIDDLEWARE
// ============================================

const { logger, devLogger, prodLogger } = require('./middleware/logger');
const { errorHandler, notFoundHandler } = require('./middleware/error_handler');
const { compression, devCompression, prodCompression } = require('./middleware/compression');

// ============================================
// IMPORT ROUTES
// ============================================

const routes = require('./routes');

// ============================================
// CREATE EXPRESS APP
// ============================================

const app = express();

// Set trust proxy for proper IP detection behind reverse proxies
app.set('trust proxy', true);

// Disable x-powered-by header for security
app.disable('x-powered-by');

// ============================================
// MIDDLEWARE STACK
// ============================================

console.log(`[Server] Starting in ${NODE_ENV} mode...`);

// 1. Compression (apply first for maximum efficiency)
if (NODE_ENV === 'production') {
    app.use(prodCompression());
    console.log('[Server] Production compression enabled');
} else {
    app.use(devCompression());
    console.log('[Server] Development compression enabled');
}

// 2. Request logging
if (NODE_ENV === 'production') {
    app.use(prodLogger());
    console.log('[Server] Production logging enabled');
} else {
    app.use(devLogger());
    console.log('[Server] Development logging enabled');
}

// 3. Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
console.log('[Server] Body parsing middleware enabled');

// 4. Security headers
app.use((req, res, next) => {
    // Basic security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');
    res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');
    
    // Content Security Policy (adjust as needed)
    if (NODE_ENV === 'production') {
        res.setHeader(
            'Content-Security-Policy',
            "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://d3js.org; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; img-src 'self' data: blob:; font-src 'self' https://cdn.jsdelivr.net;"
        );
    }
    
    next();
});
console.log('[Server] Security headers enabled');

// 5. Static file serving
const publicPath = path.join(__dirname, 'public');
app.use(express.static(publicPath, {
    maxAge: NODE_ENV === 'production' ? '1d' : 0,
    etag: true,
    lastModified: true,
    index: false // We'll handle index routing manually
}));
console.log(`[Server] Static files served from: ${publicPath}`);

// 6. Application routes
app.use('/', routes);
console.log('[Server] Application routes registered');

// 7. 404 handler (must be after all routes)
app.use(notFoundHandler);

// 8. Global error handler (must be last)
app.use(errorHandler);
console.log('[Server] Error handlers registered');

// ============================================
// HEALTH CHECK ENDPOINT
// ============================================

app.get('/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'corruption-reporting-frontend',
        environment: NODE_ENV,
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        backend: BACKEND_URL
    });
});

// ============================================
// SERVER STARTUP
// ============================================

let server;

function startServer() {
    return new Promise((resolve, reject) => {
        // Check if port is available
        const testServer = require('net').createServer();
        
        testServer.once('error', (err) => {
            if (err.code === 'EADDRINUSE') {
                console.error(`[Server] ERROR: Port ${PORT} is already in use`);
                reject(err);
            } else {
                reject(err);
            }
        });
        
        testServer.once('listening', () => {
            testServer.close();
            
            // Start Express server
            server = app.listen(PORT, HOST, () => {
                console.log('\n==============================================');
                console.log('  Corruption Reporting System - Frontend');
                console.log('==============================================');
                console.log(`Environment:    ${NODE_ENV}`);
                console.log(`Server:         http://${HOST === '0.0.0.0' ? 'localhost' : HOST}:${PORT}`);
                console.log(`Backend:        ${BACKEND_URL}`);
                console.log(`Process ID:     ${process.pid}`);
                console.log(`Node Version:   ${process.version}`);
                console.log(`Memory Usage:   ${Math.round(process.memoryUsage().heapUsed / 1024 / 1024)}MB`);
                console.log('==============================================\n');
                console.log('[Server] Ready to accept connections');
                
                resolve(server);
            });
            
            // Handle server errors
            server.on('error', (error) => {
                console.error('[Server] Server error:', error);
                reject(error);
            });
        });
        
        testServer.listen(PORT, HOST);
    });
}

// ============================================
// GRACEFUL SHUTDOWN
// ============================================

function gracefulShutdown(signal) {
    console.log(`\n[Server] Received ${signal}, starting graceful shutdown...`);
    
    if (!server) {
        console.log('[Server] No active server to shutdown');
        process.exit(0);
        return;
    }
    
    // Stop accepting new connections
    server.close((err) => {
        if (err) {
            console.error('[Server] Error during shutdown:', err);
            process.exit(1);
        }
        
        console.log('[Server] Server closed successfully');
        console.log('[Server] All connections closed');
        
        // Cleanup tasks
        console.log('[Server] Running cleanup tasks...');
        
        // Close any open resources here
        // (database connections, file handles, etc.)
        
        console.log('[Server] Cleanup complete');
        console.log('[Server] Shutdown complete\n');
        process.exit(0);
    });
    
    // Force shutdown after 10 seconds
    setTimeout(() => {
        console.error('[Server] Forceful shutdown after timeout');
        process.exit(1);
    }, 10000);
}

// ============================================
// SIGNAL HANDLERS
// ============================================

// Handle termination signals
process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('[Server] Uncaught Exception:', error);
    console.error(error.stack);
    gracefulShutdown('uncaughtException');
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error('[Server] Unhandled Rejection at:', promise);
    console.error('[Server] Reason:', reason);
    // Don't exit on unhandled rejection in production
    if (NODE_ENV !== 'production') {
        gracefulShutdown('unhandledRejection');
    }
});

// Handle process warnings
process.on('warning', (warning) => {
    console.warn('[Server] Warning:', warning.name);
    console.warn('[Server] Message:', warning.message);
    console.warn('[Server] Stack:', warning.stack);
});

// ============================================
// START SERVER
// ============================================

// Only start server if this file is run directly
if (require.main === module) {
    startServer().catch((error) => {
        console.error('[Server] Failed to start server:', error);
        process.exit(1);
    });
}

// ============================================
// EXPORTS
// ============================================

module.exports = {
    app,
    server,
    startServer,
    gracefulShutdown
};
