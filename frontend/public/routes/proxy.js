/**
 * Corruption Reporting System - API Proxy Middleware
 * Version: 1.0.0
 * Description: Proxies API requests from frontend to Python FastAPI backend
 * 
 * This middleware forwards all /api/* requests to the Python backend server,
 * handling authentication, file uploads, error responses, and request/response transformations.
 */

const express = require('express');
const axios = require('axios');

const router = express.Router();

// ============================================
// CONFIGURATION
// ============================================
const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8080';
const PROXY_TIMEOUT = parseInt(process.env.PROXY_TIMEOUT || '300000', 10); // 5 minutes for ML processing
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '0', 10); // No retries by default

// Axios instance with custom configuration
const backendClient = axios.create({
    baseURL: BACKEND_URL,
    timeout: PROXY_TIMEOUT,
    maxContentLength: 100 * 1024 * 1024, // 100MB
    maxBodyLength: 100 * 1024 * 1024, // 100MB
    validateStatus: () => true // Don't throw on any status code
});

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Get headers to forward to backend
 * @param {Object} headers - Original request headers
 * @returns {Object} Filtered headers
 */
function getForwardHeaders(headers) {
    const forwardHeaders = {};

    // Headers to forward
    const allowedHeaders = [
        'content-type',
        'content-length',
        'accept',
        'accept-encoding',
        'user-agent',
        'x-forwarded-for',
        'x-real-ip',
        'x-request-id'
    ];

    allowedHeaders.forEach(header => {
        if (headers[header]) {
            forwardHeaders[header] = headers[header];
        }
    });

    return forwardHeaders;
}

/**
 * Log proxy request
 * @param {string} method - HTTP method
 * @param {string} path - Request path
 * @param {number} status - Response status
 * @param {number} duration - Request duration in ms
 */
function logProxyRequest(method, path, status, duration) {
    const timestamp = new Date().toISOString();
    const statusEmoji = status < 400 ? '' : '';
    console.log(`[${timestamp}] ${statusEmoji} PROXY ${method} ${path} → ${status} (${duration}ms)`);
}

/**
 * Handle proxy error
 * @param {Error} error - Axios error
 * @param {Object} req - Express request
 * @param {Object} res - Express response
 */
function handleProxyError(error, req, res) {
    console.error('Proxy error:', {
        message: error.message,
        code: error.code,
        path: req.path,
        method: req.method
    });

    // If backend returned a response (4xx, 5xx), forward it
    if (error.response) {
        console.error(`[Proxy] Backend returned status ${error.response.status}`);
        return res.status(error.response.status).json(error.response.data);
    }

    // Connection errors
    if (error.code === 'ECONNREFUSED') {
        return res.status(503).json({
            error: 'Backend service unavailable',
            message: 'Cannot connect to the backend server. Please ensure it is running.',
            code: 'SERVICE_UNAVAILABLE',
            timestamp: new Date().toISOString()
        });
    }

    // Timeout errors
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
        return res.status(504).json({
            error: 'Request timeout',
            message: 'The backend server took too long to respond. Please try again.',
            code: 'GATEWAY_TIMEOUT',
            timestamp: new Date().toISOString()
        });
    }

    // Network errors
    if (error.code === 'ENOTFOUND' || error.code === 'EAI_AGAIN') {
        return res.status(502).json({
            error: 'Bad gateway',
            message: 'Cannot reach the backend server.',
            code: 'BAD_GATEWAY',
            timestamp: new Date().toISOString()
        });
    }

    // Generic error
    return res.status(500).json({
        error: 'Proxy error',
        message: error.message || 'An error occurred while proxying the request',
        code: 'PROXY_ERROR',
        timestamp: new Date().toISOString()
    });
}

// ============================================
// HEALTH CHECK ENDPOINT
// ============================================
// ============================================
// HEALTH CHECK ENDPOINT
// ============================================
router.get('/v1/health', async (req, res) => {
    const startTime = Date.now();
    const targetUrl = '/api/v1/health';

    try {
        const response = await backendClient.get(targetUrl);
        const duration = Date.now() - startTime;

        logProxyRequest('GET', targetUrl, response.status, duration);

        // Add proxy information
        const healthData = response.data;
        healthData.proxy = {
            frontend_url: `http://localhost:${process.env.PORT || 3000}`,
            backend_url: BACKEND_URL,
            latency_ms: duration
        };

        return res.status(response.status).json(healthData);
    } catch (error) {
        const duration = Date.now() - startTime;
        console.error(`[Proxy] Failed to connect to backend at ${BACKEND_URL}${targetUrl}: ${error.message}`);
        logProxyRequest('GET', targetUrl, 503, duration);
        return handleProxyError(error, req, res);
    }
});

// ============================================
// SUBMISSION ENDPOINTS
// ============================================

// POST /api/v1/submissions - Create new submission
router.post('/v1/submissions', async (req, res) => {
    const startTime = Date.now();

    try {
        // For multipart/form-data, forward the raw body
        const isMultipart = req.headers['content-type']?.includes('multipart/form-data');

        const response = await backendClient.post('/api/v1/submissions', req.body, {
            headers: getForwardHeaders(req.headers),
            // For file uploads, we need to handle this specially
            ...(isMultipart && { maxBodyLength: Infinity, maxContentLength: Infinity })
        });

        const duration = Date.now() - startTime;
        logProxyRequest('POST', '/api/v1/submissions', response.status, duration);

        return res.status(response.status).json(response.data);
    } catch (error) {
        const duration = Date.now() - startTime;
        logProxyRequest('POST', '/api/v1/submissions', 500, duration);
        return handleProxyError(error, req, res);
    }
});

// GET /api/v1/submissions - List submissions
router.get('/v1/submissions', async (req, res) => {
    const startTime = Date.now();

    try {
        const response = await backendClient.get('/api/v1/submissions', {
            params: req.query,
            headers: getForwardHeaders(req.headers)
        });

        const duration = Date.now() - startTime;
        logProxyRequest('GET', '/api/v1/submissions', response.status, duration);

        return res.status(response.status).json(response.data);
    } catch (error) {
        const duration = Date.now() - startTime;
        logProxyRequest('GET', '/api/v1/submissions', 500, duration);
        return handleProxyError(error, req, res);
    }
});

// GET /api/v1/submissions/:id - Get submission by ID
router.get('/v1/submissions/:id', async (req, res) => {
    const startTime = Date.now();
    const submissionId = req.params.id;

    try {
        const response = await backendClient.get(`/api/v1/submissions/${submissionId}`, {
            headers: getForwardHeaders(req.headers)
        });

        const duration = Date.now() - startTime;
        logProxyRequest('GET', `/api/v1/submissions/${submissionId}`, response.status, duration);

        return res.status(response.status).json(response.data);
    } catch (error) {
        const duration = Date.now() - startTime;
        logProxyRequest('GET', `/api/v1/submissions/${submissionId}`, 500, duration);
        return handleProxyError(error, req, res);
    }
});

// ============================================
// COUNTER-EVIDENCE ENDPOINT
// ============================================

// POST /api/v1/counter-evidence - Submit counter-evidence
router.post('/v1/counter-evidence', async (req, res) => {
    const startTime = Date.now();

    try {
        const response = await backendClient.post('/api/v1/counter-evidence', req.body, {
            headers: getForwardHeaders(req.headers)
        });

        const duration = Date.now() - startTime;
        logProxyRequest('POST', '/api/v1/counter-evidence', response.status, duration);

        return res.status(response.status).json(response.data);
    } catch (error) {
        const duration = Date.now() - startTime;
        logProxyRequest('POST', '/api/v1/counter-evidence', 500, duration);
        return handleProxyError(error, req, res);
    }
});

// ============================================
// REPORT ENDPOINT
// ============================================

// GET /api/v1/reports/:id - Download report PDF
router.get('/v1/reports/:id', async (req, res) => {
    const startTime = Date.now();
    const submissionId = req.params.id;

    try {
        const response = await backendClient.get(`/api/v1/reports/${submissionId}`, {
            headers: getForwardHeaders(req.headers),
            responseType: 'arraybuffer' // Handle binary PDF data
        });

        const duration = Date.now() - startTime;
        logProxyRequest('GET', `/api/v1/reports/${submissionId}`, response.status, duration);

        // Forward content-type and other headers
        if (response.headers['content-type']) {
            res.setHeader('Content-Type', response.headers['content-type']);
        }
        if (response.headers['content-disposition']) {
            res.setHeader('Content-Disposition', response.headers['content-disposition']);
        }
        if (response.headers['content-length']) {
            res.setHeader('Content-Length', response.headers['content-length']);
        }

        return res.status(response.status).send(response.data);
    } catch (error) {
        const duration = Date.now() - startTime;
        logProxyRequest('GET', `/api/v1/reports/${submissionId}`, 500, duration);
        return handleProxyError(error, req, res);
    }
});

// ============================================
// GENERIC PROXY (CATCH-ALL)
// ============================================
// This handles any API endpoints not explicitly defined above
router.all('*', async (req, res) => {
    const startTime = Date.now();

    try {
        const response = await backendClient({
            method: req.method,
            url: `/api${req.path}`,
            data: req.body,
            params: req.query,
            headers: getForwardHeaders(req.headers),
            responseType: req.path.includes('/reports/') ? 'arraybuffer' : 'json'
        });

        const duration = Date.now() - startTime;
        logProxyRequest(req.method, req.path, response.status, duration);

        // Forward response headers
        Object.keys(response.headers).forEach(key => {
            // Don't forward certain headers
            if (!['connection', 'transfer-encoding', 'content-encoding'].includes(key.toLowerCase())) {
                res.setHeader(key, response.headers[key]);
            }
        });

        return res.status(response.status).send(response.data);
    } catch (error) {
        const duration = Date.now() - startTime;
        logProxyRequest(req.method, req.path, 500, duration);
        return handleProxyError(error, req, res);
    }
});

// ============================================
// EXPORTS
// ============================================
module.exports = router;
