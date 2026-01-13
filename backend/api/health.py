"""
Health check and metrics endpoints for system monitoring.
Provides detailed status of all components and performance metrics.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict

from fastapi import APIRouter, Depends, status
from fastapi.responses import PlainTextResponse

from backend.api import schemas
from backend.api.dependencies import (
    get_storage_service,
    get_hash_chain_service,
    get_crypto_service,
    get_metrics_service,
    check_system_health
)

# Initialize logger
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Track system startup time
STARTUP_TIME = time.time()


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@router.get(
    "/health",
    response_model=schemas.HealthCheckResponse,
    summary="System health check",
    description="""
    Check health status of all system components.
    
    Returns detailed status for:
    - Storage service (read/write capability)
    - Hash chain integrity
    - Cryptography service
    - ML model availability
    - Memory usage
    
    **Status Levels:**
    - `healthy`: All components operational
    - `degraded`: Some non-critical components failing
    - `unhealthy`: Critical components failing
    """,
    responses={
        200: {"description": "System healthy or degraded"},
        503: {"description": "System unhealthy - critical failure"}
    }
)
async def health_check(
    storage_service=Depends(get_storage_service),
    hash_chain_service=Depends(get_hash_chain_service),
    crypto_service=Depends(get_crypto_service)
) -> schemas.HealthCheckResponse:
    """
    Comprehensive health check of all system components.
    
    Returns:
        HealthCheckResponse: Detailed health status
    """
    checks = {}
    
    # 1. Storage Service Health
    try:
        checks["storage_readable"] = storage_service.can_read()
        checks["storage_writable"] = storage_service.can_write()
        logger.debug("Storage health check: OK")
    except Exception as e:
        logger.error(f"Storage health check failed: {e}")
        checks["storage_readable"] = False
        checks["storage_writable"] = False
    
    # 2. Hash Chain Health
    try:
        checks["hash_chain_valid"] = hash_chain_service.verify_integrity()
        logger.debug("Hash chain health check: OK")
    except Exception as e:
        logger.error(f"Hash chain health check failed: {e}")
        checks["hash_chain_valid"] = False
    
    # 3. Crypto Service Health
    try:
        checks["crypto_operational"] = crypto_service.health_check()
        logger.debug("Crypto health check: OK")
    except Exception as e:
        logger.error(f"Crypto health check failed: {e}")
        checks["crypto_operational"] = False
    
    # 4. ML Models Health
    try:
        from backend.models.model_cache import ModelCache
        cache = ModelCache()
        checks["models_loadable"] = cache.health_check()
        logger.debug("Models health check: OK")
    except Exception as e:
        logger.error(f"Models health check failed: {e}")
        checks["models_loadable"] = False
    
    # 5. Memory Health
    try:
        import psutil
        memory = psutil.virtual_memory()
        checks["memory_ok"] = memory.percent < 90.0
        checks["memory_percent"] = memory.percent
        logger.debug(f"Memory health check: {memory.percent:.1f}% used")
    except Exception as e:
        logger.error(f"Memory health check failed: {e}")
        checks["memory_ok"] = False
        checks["memory_percent"] = None
    
    # 6. Disk Space Health
    try:
        from backend.config import get_config
        config = get_config()
        data_dir = Path(config.get("data_dir", "backend/data"))
        
        if data_dir.exists():
            disk = psutil.disk_usage(str(data_dir))
            checks["disk_ok"] = disk.percent < 90.0
            checks["disk_percent"] = disk.percent
            logger.debug(f"Disk health check: {disk.percent:.1f}% used")
        else:
            checks["disk_ok"] = False
            checks["disk_percent"] = None
    except Exception as e:
        logger.error(f"Disk health check failed: {e}")
        checks["disk_ok"] = False
        checks["disk_percent"] = None
    
    # Determine overall status
    critical_checks = [
        "storage_readable",
        "storage_writable",
        "hash_chain_valid",
        "crypto_operational",
        "memory_ok"
    ]
    
    critical_failures = [
        check for check in critical_checks 
        if not checks.get(check, False)
    ]
    
    if len(critical_failures) >= 2:
        overall_status = "unhealthy"
    elif len(critical_failures) == 1 or not checks.get("models_loadable", True):
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    # Calculate uptime
    uptime_seconds = time.time() - STARTUP_TIME
    
    # Get version from config or default
    try:
        from backend.config import get_config
        config = get_config()
        version = config.get("version", "1.0.0-mvp")
    except Exception:
        version = "1.0.0-mvp"
    
    logger.info(f"Health check completed - Status: {overall_status}")
    
    return schemas.HealthCheckResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        checks=checks,
        uptime_seconds=uptime_seconds,
        version=version
    )


# ============================================================================
# DETAILED HEALTH ENDPOINT
# ============================================================================

@router.get(
    "/health/detailed",
    response_model=Dict,
    summary="Detailed health information",
    description="""
    Extended health check with additional diagnostic information.
    Includes component-specific details and performance indicators.
    """,
    responses={
        200: {"description": "Detailed health information"}
    }
)
async def detailed_health_check(
    storage_service=Depends(get_storage_service),
    hash_chain_service=Depends(get_hash_chain_service),
    metrics_service=Depends(get_metrics_service)
) -> Dict:
    """
    Detailed health check with diagnostic information.
    
    Returns:
        dict: Extended health details
    """
    details = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "uptime_seconds": time.time() - STARTUP_TIME,
        "components": {}
    }
    
    # Storage details
    try:
        storage_stats = storage_service.get_statistics()
        details["components"]["storage"] = {
            "status": "healthy",
            "total_submissions": storage_stats.get("total_submissions", 0),
            "pending_submissions": storage_stats.get("pending_submissions", 0),
            "completed_submissions": storage_stats.get("completed_submissions", 0),
            "cache_size": storage_stats.get("cache_size", 0)
        }
    except Exception as e:
        details["components"]["storage"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Hash chain details
    try:
        chain_stats = hash_chain_service.get_statistics()
        details["components"]["hash_chain"] = {
            "status": "healthy",
            "chain_length": chain_stats.get("length", 0),
            "last_hash": chain_stats.get("last_hash", "N/A")[:16],
            "integrity": "valid" if hash_chain_service.verify_integrity() else "invalid"
        }
    except Exception as e:
        details["components"]["hash_chain"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Metrics details
    try:
        metrics = metrics_service.get_metrics()
        details["metrics"] = {
            "total_requests": metrics.get("total_requests", 0),
            "avg_response_time": metrics.get("avg_response_time", 0.0),
            "error_rate": metrics.get("error_rate", 0.0),
            "cache_hit_rate": metrics.get("cache_hit_rate", 0.0)
        }
    except Exception as e:
        details["metrics"] = {
            "error": str(e)
        }
    
    # System resources
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        details["resources"] = {
            "memory": {
                "total_mb": memory.total / (1024 * 1024),
                "used_mb": memory.used / (1024 * 1024),
                "percent": memory.percent
            },
            "cpu": {
                "percent": cpu,
                "count": psutil.cpu_count()
            }
        }
    except Exception as e:
        details["resources"] = {
            "error": str(e)
        }
    
    return details


# ============================================================================
# METRICS ENDPOINT (PROMETHEUS FORMAT)
# ============================================================================

@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus-compatible metrics",
    description="""
    Export system metrics in Prometheus text format.
    Compatible with Prometheus, Grafana, and other monitoring tools.
    """,
    responses={
        200: {
            "description": "Metrics in Prometheus format",
            "content": {"text/plain": {}}
        }
    }
)
async def prometheus_metrics(
    storage_service=Depends(get_storage_service),
    metrics_service=Depends(get_metrics_service)
) -> str:
    """
    Export metrics in Prometheus text format.
    
    Returns:
        str: Prometheus-formatted metrics
    """
    try:
        metrics_lines = []
        
        # System uptime
        uptime = time.time() - STARTUP_TIME
        metrics_lines.append(f"# HELP system_uptime_seconds System uptime in seconds")
        metrics_lines.append(f"# TYPE system_uptime_seconds gauge")
        metrics_lines.append(f"system_uptime_seconds {uptime:.2f}")
        
        # Storage metrics
        try:
            storage_stats = storage_service.get_statistics()
            
            metrics_lines.append(f"# HELP submissions_total Total number of submissions")
            metrics_lines.append(f"# TYPE submissions_total counter")
            metrics_lines.append(
                f"submissions_total {storage_stats.get('total_submissions', 0)}"
            )
            
            metrics_lines.append(f"# HELP submissions_pending Pending submissions")
            metrics_lines.append(f"# TYPE submissions_pending gauge")
            metrics_lines.append(
                f"submissions_pending {storage_stats.get('pending_submissions', 0)}"
            )
            
            metrics_lines.append(f"# HELP submissions_completed Completed submissions")
            metrics_lines.append(f"# TYPE submissions_completed gauge")
            metrics_lines.append(
                f"submissions_completed {storage_stats.get('completed_submissions', 0)}"
            )
        except Exception as e:
            logger.error(f"Failed to get storage metrics: {e}")
        
        # Request metrics
        try:
            request_metrics = metrics_service.get_metrics()
            
            metrics_lines.append(f"# HELP http_requests_total Total HTTP requests")
            metrics_lines.append(f"# TYPE http_requests_total counter")
            metrics_lines.append(
                f"http_requests_total {request_metrics.get('total_requests', 0)}"
            )
            
            metrics_lines.append(
                f"# HELP http_request_duration_seconds Average request duration"
            )
            metrics_lines.append(f"# TYPE http_request_duration_seconds gauge")
            metrics_lines.append(
                f"http_request_duration_seconds "
                f"{request_metrics.get('avg_response_time', 0.0):.6f}"
            )
            
            metrics_lines.append(f"# HELP cache_hit_rate Cache hit rate ratio")
            metrics_lines.append(f"# TYPE cache_hit_rate gauge")
            metrics_lines.append(
                f"cache_hit_rate {request_metrics.get('cache_hit_rate', 0.0):.4f}"
            )
        except Exception as e:
            logger.error(f"Failed to get request metrics: {e}")
        
        # System resource metrics
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)
            
            metrics_lines.append(f"# HELP memory_usage_bytes Memory usage in bytes")
            metrics_lines.append(f"# TYPE memory_usage_bytes gauge")
            metrics_lines.append(f"memory_usage_bytes {memory.used}")
            
            metrics_lines.append(f"# HELP memory_usage_percent Memory usage percentage")
            metrics_lines.append(f"# TYPE memory_usage_percent gauge")
            metrics_lines.append(f"memory_usage_percent {memory.percent:.2f}")
            
            metrics_lines.append(f"# HELP cpu_usage_percent CPU usage percentage")
            metrics_lines.append(f"# TYPE cpu_usage_percent gauge")
            metrics_lines.append(f"cpu_usage_percent {cpu:.2f}")
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
        
        # Join all metrics with newlines
        return "\n".join(metrics_lines) + "\n"
        
    except Exception as e:
        logger.error(f"Metrics endpoint error: {e}", exc_info=True)
        return f"# Error generating metrics: {str(e)}\n"


# ============================================================================
# READINESS PROBE ENDPOINT
# ============================================================================

@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness probe",
    description="""
    Check if system is ready to accept requests.
    Used by load balancers and orchestrators (Kubernetes).
    
    Returns 200 if ready, 503 if not ready.
    """,
    responses={
        200: {"description": "System ready"},
        503: {"description": "System not ready"}
    }
)
async def readiness_probe() -> Dict[str, str]:
    """
    Kubernetes-style readiness probe.
    
    Checks if system can handle requests.
    
    Returns:
        dict: Readiness status
    """
    try:
        # Check critical components
        health_status = await check_system_health()
        
        # Consider system ready if no critical failures
        critical_checks = [
            "storage",
            "hash_chain",
            "crypto",
            "memory"
        ]
        
        all_ready = all(health_status.get(check, False) for check in critical_checks)
        
        if all_ready:
            return {"status": "ready"}
        else:
            logger.warning("Readiness check failed - system not ready")
            from fastapi import HTTPException
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not ready"
            )
            
    except Exception as e:
        logger.error(f"Readiness probe error: {e}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Readiness check failed: {str(e)}"
        )


# ============================================================================
# LIVENESS PROBE ENDPOINT
# ============================================================================

@router.get(
    "/alive",
    status_code=status.HTTP_200_OK,
    summary="Liveness probe",
    description="""
    Check if system is alive and responding.
    Used by load balancers and orchestrators (Kubernetes).
    
    Always returns 200 if server is running.
    """,
    responses={
        200: {"description": "System alive"}
    }
)
async def liveness_probe() -> Dict[str, str]:
    """
    Kubernetes-style liveness probe.
    
    Simple check that process is running.
    
    Returns:
        dict: Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ============================================================================
# VERSION ENDPOINT
# ============================================================================

@router.get(
    "/version",
    response_model=Dict[str, str],
    summary="System version information",
    description="""
    Get system version and build information.
    """,
    responses={
        200: {"description": "Version information"}
    }
)
async def version_info() -> Dict[str, str]:
    """
    Return system version information.
    
    Returns:
        dict: Version details
    """
    try:
        from backend.config import get_config
        config = get_config()
        
        version_data = {
            "version": config.get("version", "1.0.0-mvp"),
            "environment": config.get("environment", "development"),
            "build_date": config.get("build_date", "unknown"),
            "python_version": f"{__import__('sys').version_info.major}."
                             f"{__import__('sys').version_info.minor}."
                             f"{__import__('sys').version_info.micro}"
        }
        
        return version_data
        
    except Exception as e:
        logger.error(f"Version endpoint error: {e}")
        return {
            "version": "1.0.0-mvp",
            "environment": "unknown",
            "error": str(e)
        }
