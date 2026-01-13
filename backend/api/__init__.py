"""
API package initialization.
Registers all API routers and exposes main router for FastAPI app.
"""

import logging
from fastapi import APIRouter

# Initialize logger
logger = logging.getLogger(__name__)

# Create main API router
api_router = APIRouter()


def register_routes() -> APIRouter:
    """
    Register all API routes into main router.
    
    Returns:
        APIRouter: Configured router with all endpoints
    """
    from backend.api import routes, health
    
    # Include submission and credibility routes
    api_router.include_router(
        routes.router,
        tags=["submissions"]
    )
    
    # Include health and metrics routes
    api_router.include_router(
        health.router,
        tags=["health"]
    )
    
    logger.info("All API routes registered successfully")
    
    return api_router


# Export main router
router = register_routes()

__all__ = ["router"]
