"""
FastAPI Main Application - Entry point for the corruption reporting system

Provides:
- FastAPI application setup
- Middleware configuration
- Route registration
- Startup/shutdown hooks
- Health monitoring
- Prometheus metrics integration
"""

from contextlib import asynccontextmanager
from typing import Dict, Any
import sys
import time
import psutil
from pathlib import Path

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# Prometheus client imports
from prometheus_client import (
    Counter, Gauge, Histogram, Summary,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry, REGISTRY
)

from backend.core import orchestrator
# from backend.services import queue_service, storage_service
# from backend.services import metrics_service

# Import classes, not modules
from backend.services.storage_service import StorageService
from backend.services.queue_service import QueueService
from backend.services.metrics_service import MetricsService

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
    RateLimitMiddleware,
    # metrics_middleware,
    ErrorHandlingMiddleware
)

# Import services
from backend.services.storage_service import StorageService
from backend.services.hash_chain_service import HashChainService
from backend.services.metrics_service import MetricsService

# Import workers
from backend.workers.submission_worker import SubmissionWorker
from backend.workers.cleanup_worker import CleanupWorker

# ==================== PROMETHEUS METRICS DECLARATIONS ====================

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint']
)

http_request_size_bytes = Summary(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

http_response_size_bytes = Summary(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

# Application Metrics
submission_total = Counter(
    'submission_total',
    'Total evidence submissions',
    ['status']
)

submission_processing_duration_seconds = Histogram(
    'submission_processing_duration_seconds',
    'Submission processing duration in seconds',
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

# System Resource Metrics
process_memory_percent = Gauge(
    'process_memory_percent',
    'Memory usage percentage'
)

process_cpu_percent = Gauge(
    'process_cpu_percent',
    'CPU usage percentage'
)

process_memory_bytes = Gauge(
    'process_memory_bytes',
    'Memory usage in bytes',
    ['type']
)

# Layer-Specific Metrics
layer1_anonymity_violations_total = Counter(
    'layer1_anonymity_violations_total',
    'Total anonymity violations detected'
)

layer2_credibility_score_avg = Gauge(
    'layer2_credibility_score_avg',
    'Average credibility score'
)

layer2_deepfake_detected_total = Counter(
    'layer2_deepfake_detected_total',
    'Total deepfakes detected'
)

layer3_coordination_detected_total = Counter(
    'layer3_coordination_detected_total',
    'Total coordinated attacks detected'
)

layer4_consensus_iterations_avg = Gauge(
    'layer4_consensus_iterations_avg',
    'Average consensus iterations'
)

layer4_consensus_convergence_time_seconds = Gauge(
    'layer4_consensus_convergence_time_seconds',
    'Consensus convergence time in seconds'
)

layer5_counter_evidence_total = Counter(
    'layer5_counter_evidence_total',
    'Total counter-evidence submissions'
)

layer5_counter_evidence_impact_percent = Gauge(
    'layer5_counter_evidence_impact_percent',
    'Counter-evidence impact percentage'
)

layer6_reports_generated_total = Counter(
    'layer6_reports_generated_total',
    'Total reports generated'
)

# Model Metrics
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

model_loading_failures_total = Counter(
    'model_loading_failures_total',
    'Total model loading failures',
    ['model']
)

model_cache_hit_total = Counter(
    'model_cache_hit_total',
    'Total model cache hits',
    ['model']
)

model_cache_miss_total = Counter(
    'model_cache_miss_total',
    'Total model cache misses',
    ['model']
)

# Security Metrics
hash_chain_validation_failures_total = Counter(
    'hash_chain_validation_failures_total',
    'Total hash chain validation failures'
)

validation_failures_total = Counter(
    'validation_failures_total',
    'Total input validation failures',
    ['validation_type']
)

crypto_operation_failures_total = Counter(
    'crypto_operation_failures_total',
    'Total cryptographic operation failures',
    ['operation']
)

rate_limiter_blocked_total = Counter(
    'rate_limiter_blocked_total',
    'Total requests blocked by rate limiter',
    ['reason']
)

# Storage Metrics
storage_used_percent = Gauge(
    'storage_used_percent',
    'Storage usage percentage'
)

storage_corruption_detected_total = Counter(
    'storage_corruption_detected_total',
    'Total storage corruption events detected'
)

storage_index_inconsistencies_total = Counter(
    'storage_index_inconsistencies_total',
    'Total storage index inconsistencies'
)

backup_failures_total = Counter(
    'backup_failures_total',
    'Total backup failures'
)

# Queue Metrics
queue_pending_jobs = Gauge(
    'queue_pending_jobs',
    'Number of pending jobs in queue'
)

queue_processing_jobs = Gauge(
    'queue_processing_jobs',
    'Number of jobs currently processing'
)

queue_processing_duration_seconds = Histogram(
    'queue_processing_duration_seconds',
    'Job processing duration in seconds',
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
)

queue_failures_total = Counter(
    'queue_failures_total',
    'Total queue job failures',
    ['job_type']
)

# Research/Evaluation Metrics
evaluation_auroc_score = Gauge(
    'evaluation_auroc_score',
    'Current AUROC score from evaluation'
)

evaluation_precision_score = Gauge(
    'evaluation_precision_score',
    'Current precision score from evaluation'
)

evaluation_recall_score = Gauge(
    'evaluation_recall_score',
    'Current recall score from evaluation'
)


# ==================== PROMETHEUS MIDDLEWARE ====================

async def prometheus_metrics_middleware(request: Request, call_next):
    """
    Middleware to track HTTP metrics with Prometheus.

    Tracks:
    - Request count by method, endpoint, and status
    - Request duration
    - Requests in progress
    - Request/response sizes
    """
    # Skip metrics endpoint to avoid recursion
    if request.url.path == "/metrics":
        return await call_next(request)

    method = request.method
    endpoint = request.url.path

    # Normalize endpoint (remove IDs, UUIDs)
    import re
    endpoint_normalized = re.sub(r'/[a-f0-9-]{36}', '/{id}', endpoint)
    endpoint_normalized = re.sub(r'/\d+', '/{id}', endpoint_normalized)

    # Track request size
    content_length = request.headers.get('content-length', 0)
    try:
        request_size = int(content_length)
        http_request_size_bytes.labels(method=method, endpoint=endpoint_normalized).observe(request_size)
    except (ValueError, TypeError):
        pass

    # Track requests in progress
    http_requests_in_progress.labels(method=method, endpoint=endpoint_normalized).inc()

    start_time = time.time()

    try:
        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Track metrics
        http_requests_total.labels(
            method=method,
            endpoint=endpoint_normalized,
            status=response.status_code
        ).inc()

        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint_normalized
        ).observe(duration)

        # Track response size
        if hasattr(response, 'body'):
            response_size = len(response.body)
            http_response_size_bytes.labels(
                method=method,
                endpoint=endpoint_normalized
            ).observe(response_size)

        return response

    except Exception as e:
        # Track error
        duration = time.time() - start_time

        http_requests_total.labels(
            method=method,
            endpoint=endpoint_normalized,
            status=500
        ).inc()

        http_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint_normalized
        ).observe(duration)

        raise

    finally:
        # Decrement in-progress counter
        http_requests_in_progress.labels(method=method, endpoint=endpoint_normalized).dec()


# ==================== SYSTEM METRICS UPDATER ====================

def update_system_metrics():
    """Update system resource metrics."""
    try:
        # Get process
        process = psutil.Process()

        # Memory metrics
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()

        process_memory_percent.set(memory_percent)
        process_memory_bytes.labels(type='rss').set(memory_info.rss)
        process_memory_bytes.labels(type='vms').set(memory_info.vms)

        # CPU metrics
        cpu_percent = process.cpu_percent(interval=0.1)
        process_cpu_percent.set(cpu_percent)

    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Failed to update system metrics: {e}")


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
    - Initialize Prometheus metrics
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
        storage._initialize_directories()

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
                hash_chain_validation_failures_total.inc()

        # Initialize metrics
        if config.metrics.enabled:
            logger.info("Initializing metrics service...")
            metrics = MetricsService()
            app.state.metrics = metrics

            # Link Prometheus metrics to MetricsService
            app.state.prometheus_metrics = {
                'layer2_score': layer2_credibility_score_avg,
                'layer3_coordination': layer3_coordination_detected_total,
                'layer4_iterations': layer4_consensus_iterations_avg,
                'layer4_convergence': layer4_consensus_convergence_time_seconds,
                'layer5_counter': layer5_counter_evidence_total,
                'layer5_impact': layer5_counter_evidence_impact_percent,
                'submission_total': submission_total,
                'queue_pending': queue_pending_jobs,
                'queue_processing': queue_processing_jobs,
            }

        # Start background workers
        logger.info("Starting background workers...")

        # CORRECT:
        from backend.services.queue_service import QueueService

        queue = QueueService()
        submission_worker = SubmissionWorker(
            storage_service=storage,  # Already instantiated above
            queue_service=queue,      # Now an instance, not module
            metrics_service=metrics,  # Already instantiated above
            orchestrator=orchestrator
        )

        await submission_worker.start()
        app.state.submission_worker = submission_worker

        # Cleanup worker
        if not config.testing:
            cleanup_worker = CleanupWorker(
                storage_service=storage_service,
                metrics_service=metrics_service,
                data_dir=config.storage.data_dir,
            )
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

                # Track loading failures
                for model_name, success in preload_results.items():
                    if not success:
                        model_loading_failures_total.labels(model=model_name).inc()

            except Exception as e:
                logger.warning(f"Model preloading failed: {e}")
                logger.info("Models will be loaded on first use")

        # Initialize system metrics
        logger.info("Initializing system metrics...")
        update_system_metrics()

        logger.info("Application startup complete")
        logger.info(f"Prometheus metrics available at: /metrics")

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

    # Custom middleware (order matters - error handler first)
    app.middleware("http")(ErrorHandlingMiddleware)

    # Prometheus metrics middleware
    if config.metrics.enabled:
        app.middleware("http")(prometheus_metrics_middleware)

    # Application metrics middleware
    # if config.metrics.enabled:
    #     app.middleware("http")(metrics_middleware)

    # Rate limiting
    if config.rate_limit.enabled:
        app.middleware("http")(RateLimitMiddleware)

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

        # Track validation failures
        validation_failures_total.labels(validation_type='request').inc()

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
            "docs": f"/api/{API_VERSION}/docs" if config.debug else None,
            "metrics": "/metrics" if config.metrics.enabled else None
        }

    # Prometheus metrics endpoint
    @app.get("/metrics", tags=["monitoring"])
    async def metrics() -> Response:
        """
        Prometheus metrics endpoint.

        Exposes all application metrics in Prometheus format.
        """
        # Update system metrics before export
        update_system_metrics()

        # Generate metrics
        metrics_output = generate_latest(REGISTRY)

        return Response(
            content=metrics_output,
            media_type=CONTENT_TYPE_LATEST
        )

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
    logger.info(f"Metrics available at: http://{host}:{port}/metrics")

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=config.server.log_level.lower(),
        access_log=True
    )  