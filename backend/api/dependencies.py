"""
Dependency injection functions for FastAPI routes.
Provides singleton instances of services and validation logic.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from flask import request

# Import configuration
from backend.config import get_config
from backend.services import crypto_service, hash_chain_service, metadata_service, storage_service, validation_service
from backend.utils import graph_utils, text_utils

# Initialize logger
logger = logging.getLogger(__name__)


# ============================================================================
# SERVICE SINGLETONS
# ============================================================================

@lru_cache()
def get_storage_service():
    """
    Get singleton instance of StorageService.
    
    Returns:
        StorageService: Initialized storage service instance
        
    Raises:
        RuntimeError: If storage initialization fails
    """
    try:
        from backend.services.storage_service import StorageService
        
        config = get_config()
        
        # Initialize storage service with data directory
        storage = StorageService(
            data_dir=config.storage.data_dir,
            enable_caching=True, # Default to True as not in config
            cache_ttl_seconds=300 # Default to 300 as not in config
        )
        
        logger.info("StorageService initialized successfully")
        return storage
        
    except Exception as e:
        logger.error(f"Failed to initialize StorageService: {e}", exc_info=True)
        raise RuntimeError(f"Storage service initialization failed: {e}")


@lru_cache()
def get_hash_chain_service():
    """
    Get singleton instance of HashChainService.
    
    Returns:
        HashChainService: Initialized hash chain service
        
    Raises:
        RuntimeError: If hash chain initialization fails
    """
    try:
        from backend.services.hash_chain_service import HashChainService
        
        config = get_config()
        chain_file = config.storage.data_dir / "chain.json"
        
        # Initialize and verify chain integrity
        chain_service = HashChainService(chain_file=chain_file)
        chain_service.verify_integrity()
        
        logger.info("HashChainService initialized and verified")
        return chain_service
        
    except Exception as e:
        logger.error(f"Failed to initialize HashChainService: {e}", exc_info=True)
        raise RuntimeError(f"Hash chain service initialization failed: {e}")


@lru_cache()
def get_crypto_service():
    """
    Get singleton instance of CryptoService.
    
    Returns:
        CryptoService: Initialized cryptography service
        
    Raises:
        RuntimeError: If crypto initialization fails
    """
    try:
        from backend.services.crypto_service import CryptoService
        
        # Initialize with encryption key from environment
        crypto = CryptoService()
        
        logger.info("CryptoService initialized successfully")
        return crypto
        
    except Exception as e:
        logger.error(f"Failed to initialize CryptoService: {e}", exc_info=True)
        raise RuntimeError(f"Crypto service initialization failed: {e}")


@lru_cache()
def get_rate_limiter():
    """
    Get singleton instance of RateLimiter.
    
    Returns:
        RateLimiter: Initialized rate limiter service
        
    Raises:
        RuntimeError: If rate limiter initialization fails
    """
    try:
        from backend.services.rate_limiter import RateLimiter
        
        config = get_config()
        
        # Initialize rate limiter with configuration
        # Use RateLimitConfig attributes
        rate_limiter = RateLimiter(
            default_limit=config.rate_limit.max_submissions_per_hour,
            window_seconds=3600,  # Per hour
            cleanup_interval=config.rate_limit.cleanup_interval
        )
        
        logger.info("RateLimiter initialized successfully")
        return rate_limiter
        
    except Exception as e:
        logger.error(f"Failed to initialize RateLimiter: {e}", exc_info=True)
        raise RuntimeError(f"Rate limiter initialization failed: {e}")


@lru_cache()
def get_validation_service():
    """
    Get singleton instance of ValidationService.
    
    Returns:
        ValidationService: Initialized validation service
        
    Raises:
        RuntimeError: If validation service initialization fails
    """
    try:
        from backend.services.validation_service import ValidationService
        
        config = get_config()
        
        # Initialize with size limits
        validation = ValidationService(
            max_image_size_mb=5, # Default
            max_audio_size_mb=5, # Default
            max_video_size_mb=50, # Default
            allowed_extensions=[".jpg", ".jpeg", ".png", ".wav", ".mp3", ".mp4", ".avi"]
        )
        
        logger.info("ValidationService initialized successfully")
        return validation
        
    except Exception as e:
        logger.error(f"Failed to initialize ValidationService: {e}", exc_info=True)
        raise RuntimeError(f"Validation service initialization failed: {e}")


@lru_cache()
def get_orchestrator():
    """
    Get singleton instance of Orchestrator.
    
    The orchestrator coordinates all 6 layers and manages the processing pipeline.
    
    Returns:
        Orchestrator: Initialized orchestrator instance
        
    Raises:
        RuntimeError: If orchestrator initialization fails
    """
    try:
        from backend.core.orchestrator import Orchestrator
        
        # Get all required services
        storage = get_storage_service()
        hash_chain = get_hash_chain_service()
        crypto = get_crypto_service()
        metrics_service = getattr(request.app.state, 'metrics', None)
        
        # Initialize orchestrator with services
        orchestrator = Orchestrator(
            storage_service=storage_service,
            hash_chain_service=hash_chain_service,
            crypto_service=crypto_service,
            metadata_service=metadata_service,
            validation_service=validation_service,
            text_utils=text_utils,
            graph_utils=graph_utils,
            metrics_service=metrics_service  # ADD THIS
        )
        
        logger.info("Orchestrator initialized successfully")
        return orchestrator
        
    except Exception as e:
        logger.error(f"Failed to initialize Orchestrator: {e}", exc_info=True)
        raise RuntimeError(f"Orchestrator initialization failed: {e}")


@lru_cache()
def get_metrics_service():
    """
    Get singleton instance of MetricsService.
    
    Returns:
        MetricsService: Initialized metrics tracking service
        
    Raises:
        RuntimeError: If metrics service initialization fails
    """
    try:
        from backend.services.metrics_service import MetricsService
        
        # Initialize metrics service
        metrics = MetricsService()
        
        logger.info("MetricsService initialized successfully")
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to initialize MetricsService: {e}", exc_info=True)
        raise RuntimeError(f"Metrics service initialization failed: {e}")


# ============================================================================
# VALIDATION DEPENDENCIES
# ============================================================================

async def verify_submission_exists(
    submission_id: str,
    storage_service=Depends(get_storage_service)
) -> dict:
    """
    Verify that a submission exists in storage.
    
    Args:
        submission_id: UUID of submission to verify
        storage_service: Injected storage service instance
        
    Returns:
        dict: Submission data if found
        
    Raises:
        HTTPException: 404 if submission not found
    """
    try:
        submission = storage_service.load_submission(submission_id)
        
        if not submission:
            logger.warning(f"Submission not found: {submission_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Submission {submission_id} not found"
            )
        
        return submission
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying submission {submission_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify submission existence"
        )


async def check_rate_limit(
    request: Request,
    rate_limiter=Depends(get_rate_limiter)
) -> None:
    """
    Check if request exceeds rate limit.
    
    Args:
        request: FastAPI request object
        rate_limiter: Injected rate limiter instance
        
    Raises:
        HTTPException: 429 if rate limit exceeded
    """
    # Extract client IP address
    client_ip = get_client_ip(request)
    
    try:
        # Check rate limit
        is_allowed = rate_limiter.check_limit(client_ip)
        
        if not is_allowed:
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={
                    "Retry-After": str(rate_limiter.get_retry_after(client_ip))
                }
            )
        
        logger.debug(f"Rate limit check passed for IP: {client_ip}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rate limit check error for {client_ip}: {e}")
        # On error, allow request to proceed (fail open)
        pass


async def validate_file_upload(
    file_size: int,
    filename: str,
    evidence_type: str,
    validation_service=Depends(get_validation_service)
) -> None:
    """
    Validate uploaded file meets requirements.
    
    Args:
        file_size: Size of uploaded file in bytes
        filename: Original filename
        evidence_type: Type of evidence (image/audio/video)
        validation_service: Injected validation service
        
    Raises:
        HTTPException: 400 if validation fails
    """
    try:
        # Validate file extension
        validation_service.validate_file_extension(filename, evidence_type)
        
        # Validate file size
        validation_service.validate_file_size(file_size, evidence_type)
        
        logger.debug(f"File validation passed: {filename} ({file_size} bytes)")
        
    except ValueError as e:
        logger.warning(f"File validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File validation failed"
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_client_ip(request: Request) -> str:
    """
    Extract client IP address from request.
    
    Handles X-Forwarded-For header for proxy/load balancer scenarios.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Client IP address
    """
    # Check X-Forwarded-For header (proxy/load balancer)
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # Take first IP in chain (original client)
        return forwarded_for.split(",")[0].strip()
    
    # Check X-Real-IP header (nginx)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client IP
    return request.client.host if request.client else "unknown"


def get_request_id(request: Request) -> Optional[str]:
    """
    Get or generate unique request ID for tracing.
    
    Args:
        request: FastAPI request object
        
    Returns:
        str: Request ID
    """
    # Check if request ID already set by middleware
    request_id = getattr(request.state, "request_id", None)
    
    if not request_id:
        # Generate new request ID
        import uuid
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
    
    return request_id


# ============================================================================
# HEALTH CHECK DEPENDENCY
# ============================================================================

async def check_system_health() -> dict:
    """
    Check health status of all system components.
    
    Returns:
        dict: Health check results for each component
        
    Raises:
        HTTPException: 503 if critical components unhealthy
    """
    health_status = {
        "storage": False,
        "hash_chain": False,
        "crypto": False,
        "models": False,
        "memory": False
    }
    
    try:
        # Check storage service
        storage = get_storage_service()
        health_status["storage"] = storage.health_check()
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
    
    try:
        # Check hash chain service
        hash_chain = get_hash_chain_service()
        health_status["hash_chain"] = hash_chain.health_check()
    except Exception as e:
        logger.error(f"Hash chain health check failed: {e}")
    
    try:
        # Check crypto service
        crypto = get_crypto_service()
        health_status["crypto"] = crypto.health_check()
    except Exception as e:
        logger.error(f"Crypto health check failed: {e}")
    
    try:
        # Check if models can be loaded
        from backend.models.model_cache import ModelCache
        cache = ModelCache()
        health_status["models"] = cache.health_check()
    except Exception as e:
        logger.error(f"Model health check failed: {e}")
    
    try:
        # Check memory usage
        import psutil
        memory = psutil.virtual_memory()
        health_status["memory"] = memory.percent < 90  # Less than 90% used
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
    
    # Determine overall health
    all_healthy = all(health_status.values())
    
    if not all_healthy:
        critical_components = ["storage", "crypto", "memory"]
        critical_unhealthy = any(
            not health_status[comp] for comp in critical_components
        )
        
        if critical_unhealthy:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System unhealthy - critical components failed",
                headers={"X-Health-Status": str(health_status)}
            )
    
    return health_status


# ============================================================================
# STARTUP VALIDATION
# ============================================================================

def validate_environment() -> None:
    """
    Validate environment configuration at startup.
    
    Raises:
        RuntimeError: If required environment variables missing
    """
    import os
    
    required_vars = [
        "DATA_DIR",
    ]
    
    optional_vars = {
        "RATE_LIMIT_REQUESTS": "10",
        "RATE_LIMIT_WINDOW": "3600",
        "MAX_IMAGE_SIZE_MB": "5",
        "MAX_VIDEO_SIZE_MB": "50",
        "ENABLE_RATE_LIMITING": "true"
    }
    
    # Check required variables
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    # Set defaults for optional variables
    for var, default in optional_vars.items():
        if not os.getenv(var):
            os.environ[var] = default
            logger.info(f"Set default value for {var}: {default}")
    
    logger.info("Environment validation passed")


def initialize_data_directories() -> None:
    """
    Ensure all required data directories exist.
    
    Raises:
        RuntimeError: If directory creation fails
    """
    from backend.config import get_config
    
    config = get_config()
    data_dir = config.storage.data_dir
    
    required_dirs = [
        data_dir / "submissions",
        data_dir / "evidence",
        data_dir / "reports",
        data_dir / "cache",
        data_dir / "evidence" / "2026" / "01",
    ]
    
    try:
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
        
        logger.info("Data directories initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize data directories: {e}")
        raise RuntimeError(f"Data directory initialization failed: {e}")
