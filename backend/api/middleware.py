"""
FastAPI middleware for CORS, rate limiting, logging, and error handling.
"""

import logging
import time
import traceback
from typing import Callable
from uuid import uuid4

from fastapi import Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from backend.api.dependencies import get_client_ip, get_rate_limiter

# Initialize logger
logger = logging.getLogger(__name__)


# ============================================================================
# CORS CONFIGURATION
# ============================================================================

def setup_cors(app) -> None:
    """
    Configure CORS middleware for the FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    from backend.config import get_config
    
    config = get_config()
    
    # Get CORS settings from config
    allowed_origins = config.get("cors_origins", ["http://localhost:3000"])
    allow_credentials = config.get("cors_allow_credentials", True)
    allow_methods = config.get("cors_allow_methods", ["*"])
    allow_headers = config.get("cors_allow_headers", ["*"])
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
        expose_headers=["X-Request-ID", "X-Processing-Time"],
        max_age=3600  # Cache preflight requests for 1 hour
    )
    
    logger.info(f"CORS configured with origins: {allowed_origins}")


# ============================================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================================

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to log all incoming requests and responses.
    Adds request ID and timing information.
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response from downstream handler
        """
        # Generate unique request ID
        request_id = str(uuid4())
        request.state.request_id = request_id
        
        # Get client IP
        client_ip = get_client_ip(request)
        
        # Record start time
        start_time = time.time()
        
        # Log incoming request
        logger.info(
            f"REQUEST [{request_id}] {request.method} {request.url.path} "
            f"from {client_ip}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{process_time:.3f}s"
            
            # Log response
            logger.info(
                f"RESPONSE [{request_id}] {response.status_code} "
                f"in {process_time:.3f}s"
            )
            
            return response
            
        except Exception as e:
            # Log error
            process_time = time.time() - start_time
            logger.error(
                f"ERROR [{request_id}] {type(e).__name__}: {e} "
                f"after {process_time:.3f}s",
                exc_info=True
            )
            
            # Re-raise to be handled by error middleware
            raise


# ============================================================================
# RATE LIMITING MIDDLEWARE
# ============================================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce rate limits on API requests.
    Limits based on client IP address.
    """
    
    def __init__(self, app, excluded_paths: list = None):
        """
        Initialize rate limit middleware.
        
        Args:
            app: FastAPI application
            excluded_paths: List of paths to exclude from rate limiting
        """
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/metrics",
            "/docs",
            "/openapi.json"
        ]
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        Check rate limit before processing request.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response or rate limit error
        """
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Get rate limiter instance
        try:
            rate_limiter = get_rate_limiter()
        except Exception as e:
            logger.error(f"Failed to get rate limiter: {e}", exc_info=True)
            # Fail open - allow request if rate limiter unavailable
            return await call_next(request)
        
        # Get client IP
        client_ip = get_client_ip(request)
        
        # Determine endpoint type from path
        endpoint_type = 'default'
        if '/submissions' in request.url.path:
            endpoint_type = 'submission'
        elif '/health' in request.url.path:
            endpoint_type = 'health'
        elif '/report' in request.url.path:
            endpoint_type = 'report'
        elif '/counter-evidence' in request.url.path:
            endpoint_type = 'counter_evidence'
        
        # Check rate limit using correct API
        try:
            is_allowed, rate_info = rate_limiter.check_rate_limit(
                ip_address=client_ip,
                endpoint_type=endpoint_type
            )
            
            if not is_allowed:
                # Rate limit exceeded
                retry_after = rate_info.get('retry_after', 60)
                limit = rate_info.get('limit', 10)
                current = rate_info.get('current', limit)
                
                logger.warning(
                    f"Rate limit exceeded for {client_ip} on {request.url.path} "
                    f"({current}/{limit} for {endpoint_type})"
                )
                
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": retry_after,
                        "limit": limit,
                        "endpoint_type": endpoint_type
                    },
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(limit),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                    }
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers to response
            limit = rate_info.get('limit', 10)
            remaining = rate_info.get('remaining', 0)
            reset_time = rate_info.get('reset', int(time.time()) + 3600)
            
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limit check error for {client_ip}: {e}", exc_info=True)
            # Fail open - allow request on error
            return await call_next(request)



# ============================================================================
# ERROR HANDLING MIDDLEWARE
# ============================================================================

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to catch and format all unhandled exceptions.
    Provides consistent error responses.
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        Catch unhandled exceptions and return formatted error.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Normal response or error response
        """
        try:
            return await call_next(request)
            
        except Exception as e:
            # Get request ID if available
            request_id = getattr(request.state, "request_id", "unknown")
            
            # Log error with full traceback
            logger.error(
                f"Unhandled exception [{request_id}]: {type(e).__name__}: {e}",
                exc_info=True
            )
            
            # Determine error type and status code
            if isinstance(e, ValueError):
                status_code = status.HTTP_400_BAD_REQUEST
                error_type = "ValidationError"
            elif isinstance(e, FileNotFoundError):
                status_code = status.HTTP_404_NOT_FOUND
                error_type = "NotFound"
            elif isinstance(e, PermissionError):
                status_code = status.HTTP_403_FORBIDDEN
                error_type = "Forbidden"
            elif isinstance(e, TimeoutError):
                status_code = status.HTTP_504_GATEWAY_TIMEOUT
                error_type = "Timeout"
            else:
                status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
                error_type = "InternalServerError"
            
            # Build error response
            from datetime import datetime
            
            error_response = {
                "error": error_type,
                "message": str(e),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "request_id": request_id,
                "path": request.url.path
            }
            
            # Add traceback in development mode
            try:
                from backend.config import get_config
                config = get_config()
                
                if config.environment == "development":
                    error_response["traceback"] = traceback.format_exc()
            except Exception:
                # If config fails, don't add traceback
                pass
            
            return JSONResponse(
                status_code=status_code,
                content=error_response,
                headers={"X-Request-ID": request_id}
            )


# ============================================================================
# COMPRESSION MIDDLEWARE
# ============================================================================

class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Middleware to compress responses using gzip.
    Only compresses responses larger than 1KB.
    """
    
    def __init__(self, app, minimum_size: int = 1024):
        """
        Initialize compression middleware.
        
        Args:
            app: FastAPI application
            minimum_size: Minimum response size to compress (bytes)
        """
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        Compress response if client accepts gzip.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Compressed or uncompressed response
        """
        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        
        if "gzip" not in accept_encoding.lower():
            return await call_next(request)
        
        # Get response
        response = await call_next(request)
        
        # Only compress if response is large enough
        content_length = response.headers.get("Content-Length")
        
        if content_length and int(content_length) < self.minimum_size:
            return response
        
        # Check if already compressed
        if response.headers.get("Content-Encoding"):
            return response
        
        # Check content type is compressible
        content_type = response.headers.get("Content-Type", "")
        compressible_types = [
            "application/json",
            "text/html",
            "text/plain",
            "text/css",
            "application/javascript",
            "text/javascript"
        ]
        
        if not any(ct in content_type for ct in compressible_types):
            return response
        
        # Compression handled by FastAPI's GZipMiddleware
        # This middleware just ensures proper headers
        response.headers["Vary"] = "Accept-Encoding"
        
        return response


# ============================================================================
# SECURITY HEADERS MIDDLEWARE
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.
    """
    
    async def dispatch(
        self, 
        request: Request, 
        call_next: Callable
    ) -> Response:
        """
        Add security headers to response.
        
        Args:
            request: Incoming request
            call_next: Next middleware/route handler
            
        Returns:
            Response: Response with security headers
        """
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Content Security Policy (restrictive for API)
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "frame-ancestors 'none'"
        )
        
        # Remove server header for security
        response.headers.pop("Server", None)
        
        return response


# ============================================================================
# MIDDLEWARE SETUP FUNCTION
# ============================================================================

def setup_middleware(app) -> None:
    """
    Configure all middleware for the FastAPI application.
    Order matters: earliest added runs first on request, last on response.
    
    Args:
        app: FastAPI application instance
    """
    from backend.config import get_config
    
    config = get_config()
    
    # 1. Security headers (outermost - runs first/last)
    app.add_middleware(SecurityHeadersMiddleware)
    
    # 2. Error handling (catch all errors)
    app.add_middleware(ErrorHandlingMiddleware)
    
    # 3. Compression (compress outgoing responses)
    if config.get("enable_compression", True):
        # Use FastAPI's built-in GZipMiddleware
        from fastapi.middleware.gzip import GZipMiddleware
        app.add_middleware(GZipMiddleware, minimum_size=1024)
    
    # 4. Rate limiting (before processing request)
    if config.get("enable_rate_limiting", True):
        app.add_middleware(
            RateLimitMiddleware,
            excluded_paths=["/health", "/metrics", "/docs", "/openapi.json"]
        )
    
    # 5. Request logging (log all requests)
    app.add_middleware(RequestLoggingMiddleware)
    
    # 6. CORS (allow cross-origin requests)
    setup_cors(app)
    
    logger.info("All middleware configured successfully")
