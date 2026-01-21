/**
 * Corruption Reporting System - Route Registry
 * Version: 1.0.0
 * Description: Central route registration for Express server
 * 
 * This file registers all routes in the correct order:
 * 1. API proxy routes (forward to Python backend)
 * 2. Static file routes (HTML, CSS, JS)
 * 3. Error handlers
 */

const express = require('express');
const path = require('path');

/**
 * Configure and register all application routes
 * @param {express.Application} app - Express application instance
 * @returns {void}
 */
function registerRoutes(app) {
    // ============================================
    // API PROXY ROUTES
    // ============================================
    // Proxy all /api/* requests to Python backend
    // This must come before static routes to ensure API calls are intercepted
    const proxyRouter = require('./proxy');
    app.use('/api', proxyRouter);

    // ============================================
    // HEALTH CHECK (Local Frontend Health)
    // ============================================
    app.get('/health', (req, res) => {
        res.status(200).json({
            status: 'healthy',
            service: 'frontend',
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            memory: {
                used: Math.round(process.memoryUsage().heapUsed / 1024 / 1024),
                total: Math.round(process.memoryUsage().heapTotal / 1024 / 1024),
                unit: 'MB'
            }
        });
    });

    // ============================================
    // STATIC FILE ROUTES
    // ============================================
    // Serve static files (HTML, CSS, JS, images)
    const staticRouter = require('./static');
    app.use('/', staticRouter);

    // ============================================
    // EXPLICIT HTML ROUTES
    // ============================================
    // Define explicit routes for HTML pages for better SEO and routing control
    const publicDir = path.join(__dirname, '..', 'public');

    // Main submission page
    app.get('/', (req, res) => {
        res.sendFile(path.join(publicDir, 'index.html'));
    });

    app.get('/index.html', (req, res) => {
        res.sendFile(path.join(publicDir, 'index.html'));
    });

    // Review/dashboard page
    app.get('/review', (req, res) => {
        res.sendFile(path.join(publicDir, 'review.html'));
    });

    app.get('/review.html', (req, res) => {
        res.sendFile(path.join(publicDir, 'review.html'));
    });

    // Counter-evidence submission page
    app.get('/counter', (req, res) => {
        res.sendFile(path.join(publicDir, 'counter.html'));
    });

    app.get('/counter.html', (req, res) => {
        res.sendFile(path.join(publicDir, 'counter.html'));
    });

    // Error page
    app.get('/error', (req, res) => {
        res.sendFile(path.join(publicDir, 'error.html'));
    });

    app.get('/error.html', (req, res) => {
        res.sendFile(path.join(publicDir, 'error.html'));
    });

    // Help page
    app.get('/help', (req, res) => {
        res.sendFile(path.join(publicDir, 'help.html'));
    });

    app.get('/help.html', (req, res) => {
        res.sendFile(path.join(publicDir, 'help.html'));
    });

    // ============================================
    // 404 NOT FOUND HANDLER
    // ============================================
    // Must come after all valid routes
    app.use((req, res, next) => {
        // Check if it's an API request
        if (req.path.startsWith('/api')) {
            return res.status(404).json({
                error: 'API endpoint not found',
                path: req.path,
                method: req.method,
                timestamp: new Date().toISOString()
            });
        }

        // For non-API requests, redirect to error page
        const errorUrl = `/error.html?code=404&message=${encodeURIComponent('Page not found')}&endpoint=${encodeURIComponent(req.path)}`;
        res.redirect(errorUrl);
    });

    // ============================================
    // GLOBAL ERROR HANDLER
    // ============================================
    // Catches all errors that weren't handled by route-specific handlers
    app.use((err, req, res, next) => {
        console.error('Global error handler:', err);

        // Determine if it's an API request
        const isApiRequest = req.path.startsWith('/api');

        // Log error details
        const errorDetails = {
            message: err.message,
            stack: process.env.NODE_ENV === 'development' ? err.stack : undefined,
            path: req.path,
            method: req.method,
            timestamp: new Date().toISOString()
        };

        if (isApiRequest) {
            // Return JSON error for API requests
            return res.status(err.status || 500).json({
                error: err.message || 'Internal server error',
                details: errorDetails,
                status: err.status || 500
            });
        }

        // Redirect to error page for web requests
        const errorUrl = `/error.html?code=${err.status || 500}&message=${encodeURIComponent(err.message || 'Internal server error')}&details=${encodeURIComponent(JSON.stringify(errorDetails))}`;
        res.redirect(errorUrl);
    });
}

/**
 * Initialize routes with middleware
 * @param {express.Application} app - Express application instance
 * @returns {void}
 */
function initializeRoutes(app) {
    // Log all requests in development
    if (process.env.NODE_ENV === 'development') {
        app.use((req, res, next) => {
            console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
            next();
        });
    }

    // Register all routes
    registerRoutes(app);

    console.log(' Routes registered successfully');
}

/**
 * Get route information for debugging
 * @param {express.Application} app - Express application instance
 * @returns {Array} List of registered routes
 */
function getRegisteredRoutes(app) {
    const routes = [];

    // Function to extract routes from app layers
    function extractRoutes(stack, prefix = '') {
        stack.forEach((layer) => {
            if (layer.route) {
                // Regular route
                const path = prefix + layer.route.path;
                const methods = Object.keys(layer.route.methods).join(', ').toUpperCase();
                routes.push({ path, methods });
            } else if (layer.name === 'router') {
                // Router middleware
                const routerPrefix = prefix + (layer.regexp.source.match(/^\\\/([^\\]+)/) || ['', ''])[1].replace(/\\\//g, '/');
                if (layer.handle.stack) {
                    extractRoutes(layer.handle.stack, routerPrefix);
                }
            }
        });
    }

    if (app._router && app._router.stack) {
        extractRoutes(app._router.stack);
    }

    return routes;
}

/**
 * Print registered routes to console (for debugging)
 * @param {express.Application} app - Express application instance
 * @returns {void}
 */
function printRoutes(app) {
    console.log('\n=== Registered Routes ===');
    const routes = getRegisteredRoutes(app);
    
    if (routes.length === 0) {
        console.log('No routes registered');
        return;
    }

    routes.forEach(route => {
        console.log(`${route.methods.padEnd(10)} ${route.path}`);
    });
    
    console.log('========================\n');
}

// ============================================
// ROUTE DOCUMENTATION
// ============================================
/**
 * Route Structure:
 * 
 * /api/*                   - Proxied to Python backend (port 8080)
 *   /api/v1/health         - Backend health check
 *   /api/v1/submissions    - Evidence submissions (POST, GET)
 *   /api/v1/submissions/:id - Get submission by ID (GET)
 *   /api/v1/counter-evidence - Counter-evidence submission (POST)
 *   /api/v1/reports/:id    - Download report PDF (GET)
 * 
 * /health                  - Frontend health check
 * 
 * /                        - Main submission page (index.html)
 * /index.html              - Main submission page
 * /review                  - Review dashboard (review.html)
 * /review.html             - Review dashboard
 * /counter                 - Counter-evidence page (counter.html)
 * /counter.html            - Counter-evidence page
 * /error                   - Error display page (error.html)
 * /error.html              - Error display page
 * /help                    - Help documentation (help.html)
 * /help.html               - Help documentation
 * 
 * /css/*                   - CSS stylesheets
 * /js/*                    - JavaScript files
 * /assets/*                - Images, icons, etc.
 * 
 * 404                      - Redirects to /error.html?code=404
 * 500                      - Error handler
 */

// ============================================
// EXPORTS
// ============================================
module.exports = {
    registerRoutes,
    initializeRoutes,
    getRegisteredRoutes,
    printRoutes
};

