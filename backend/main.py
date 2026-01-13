"""
FastAPI Main Application - Entry point for the corruption reporting system

Provides:
- FastAPI application setup
- Middleware configuration
- Route registration
- Startup/shutdown hooks
- Health monitoring
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import sys
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import load_config, get_config
from backend.logging_config import setup_logging, get_logger
from backend.exceptions import BaseApplicationException
from backend.constants import APP_NAME, APP_VERSION, APP_DESCRIPTION, API_VERSION

# Import routers
from backend.api.routes import router as api_router
from backend.api.health import router as health_router
from backend.api.middleware import (
    rate_limit_middleware,
    metrics_middleware,
    error_handler_middleware
)

# Import services
from backend.services.storage_service import StorageService
from backend.services.hash_chain_service import HashChainService
from backend.services.metrics_service import MetricsService

# Import workers
from backend.workers.submission_worker import SubmissionWorker
from backend.workers.cleanup_worker import CleanupWorker


# ==================== APPLICATION LIFECYCLE ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown tasks:
    - Initialize services
    - Verify storage integrity
    - Start background workers
    - Preload ML models (optional)
    - Cleanup on shutdown
    """
    logger = get_logger(__name__)
    config = get_config()
    
    # ========== STARTUP ==========
    logger.info(f"Starting {APP_NAME} v{APP_VERSION}")
    logger.info(f"Environment: {config.environment}")
    
    try:
        # Initialize storage
        logger.info("Initializing storage...")
        storage = StorageService()
        storage.initialize()
        
        # Verify hash chain integrity
        if config.security.hash_chain_enabled:
            logger.info("Verifying hash chain integrity...")
            hash_chain = HashChainService()
            
            try:
                hash_chain.verify_chain()
                logger.info("Hash chain verification successful")
            except Exception as e:
                logger.warning(f"Hash chain verification failed: {e}")
                logger.info("Initializing new hash chain...")
                hash_chain.initialize()
        
        # Initialize metrics
        if config.metrics.enabled:
            logger.info("Initializing metrics service...")
            metrics = MetricsService()
            app.state.metrics = metrics
        
        # Start background workers
        logger.info("Starting background workers...")
        
        # Submission worker
        submission_worker = SubmissionWorker()
        submission_worker.start()
        app.state.submission_worker = submission_worker
        
        # Cleanup worker
        if not config.testing:
            cleanup_worker = CleanupWorker()
            cleanup_worker.start()
            app.state.cleanup_worker = cleanup_worker
        
        # Preload models (optional, can be slow)
        if not config.models.lazy_loading and not config.testing:
            logger.info("Preloading ML models (this may take a few minutes)...")
            try:
                from backend.models import preload_models
                preload_results = preload_models()
                
                loaded = sum(1 for success in preload_results.values() if success)
                logger.info(f"Preloaded {loaded}/{len(preload_results)} models")
            except Exception as e:
                logger.warning(f"Model preloading failed: {e}")
                logger.info("Models will be loaded on first use")
        
        logger.info("Application startup complete")
        
        # Yield control to application
        yield
        
        # ========== SHUTDOWN ==========
        logger.info("Shutting down application...")
        
        # Stop workers
        if hasattr(app.state, 'submission_worker'):
            logger.info("Stopping submission worker...")
            app.state.submission_worker.stop()
        
        if hasattr(app.state, 'cleanup_worker'):
            logger.info("Stopping cleanup worker...")
            app.state.cleanup_worker.stop()
        
        # Cleanup models
        logger.info("Unloading ML models...")
        try:
            from backend.models import unload_all_models
            unload_all_models()
        except Exception as e:
            logger.warning(f"Model cleanup failed: {e}")
        
        # Final metrics export
        if hasattr(app.state, 'metrics'):
            logger.info("Exporting final metrics...")
            try:
                metrics_summary = app.state.metrics.get_summary()
                logger.info(f"Final metrics: {metrics_summary}")
            except Exception as e:
                logger.warning(f"Metrics export failed: {e}")
        
        logger.info("Application shutdown complete")
    
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


# ==================== APPLICATION FACTORY ====================

def create_app(
    config_file: str = None,
    environment: str = None
) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config_file: Path to configuration file
        environment: Environment name (dev, prod)
        
    Returns:
        FastAPI: Configured application
    """
    # Load configuration
    config = load_config(config_file, environment)
    
    # Setup logging
    setup_logging(
        level=config.logging.level,
        console=config.logging.console_logging,
        file_logging=config.logging.file_logging
    )
    
    logger = get_logger(__name__)
    logger.info(f"Creating FastAPI application: environment={config.environment}")
    
    # Create FastAPI app
    app = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        debug=config.debug,
        lifespan=lifespan,
        docs_url=f"/api/{API_VERSION}/docs" if config.debug else None,
        redoc_url=f"/api/{API_VERSION}/redoc" if config.debug else None,
        openapi_url=f"/api/{API_VERSION}/openapi.json" if config.debug else None
    )
    
    # Store config in app state
    app.state.config = config
    
    # ========== MIDDLEWARE ==========
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors.allowed_origins,
        allow_credentials=config.cors.allow_credentials,
        allow_methods=config.cors.allow_methods,
        allow_headers=config.cors.allow_headers
    )
    
    # Gzip compression
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    
    # Custom middleware
    app.middleware("http")(error_handler_middleware)
    
    if config.metrics.enabled:
        app.middleware("http")(metrics_middleware)
    
    if config.rate_limit.enabled:
        app.middleware("http")(rate_limit_middleware)
    
    # ========== EXCEPTION HANDLERS ==========
    
    @app.exception_handler(BaseApplicationException)
    async def application_exception_handler(
        request: Request,
        exc: BaseApplicationException
    ) -> JSONResponse:
        """Handle application exceptions."""
        logger.warning(f"Application exception: {exc}")
        
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict()
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError
    ) -> JSONResponse:
        """Handle validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "VALIDATION_ERROR",
                "message": "Invalid request data",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(
        request: Request,
        exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.error(f"Unexpected error: {exc}", exc_info=True)
        
        if config.debug:
            import traceback
            error_detail = {
                "error": "INTERNAL_ERROR",
                "message": str(exc),
                "traceback": traceback.format_exc()
            }
        else:
            error_detail = {
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred"
            }
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=error_detail
        )
    
    # ========== ROUTES ==========
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root() -> Dict[str, Any]:
        """Root endpoint."""
        return {
            "name": APP_NAME,
            "version": APP_VERSION,
            "status": "running",
            "environment": config.environment,
            "api_version": API_VERSION,
            "docs": f"/api/{API_VERSION}/docs" if config.debug else None
        }
    
    # Health check
    app.include_router(health_router, prefix=f"/api/{API_VERSION}", tags=["health"])
    
    # Main API routes
    app.include_router(api_router, prefix=f"/api/{API_VERSION}", tags=["api"])
    
    logger.info("FastAPI application created successfully")
    
    return app


# ==================== APPLICATION INSTANCE ====================

# Create default application instance
app = create_app()


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    """
    Run application with uvicorn.
    
    Usage:
        python -m backend.main
        python -m backend.main --environment production
    """
    import argparse
    import uvicorn
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the corruption reporting system")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--environment",
        type=str,
        choices=["development", "production", "testing"],
        help="Environment name"
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, args.environment)
    
    # Override with command line arguments
    host = args.host or config.server.host
    port = args.port or config.server.port
    reload = args.reload or config.server.reload
    
    # Run server
    logger = get_logger(__name__)
    logger.info(f"Starting server: http://{host}:{port}")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=config.server.log_level.lower(),
        access_log=True
    )
