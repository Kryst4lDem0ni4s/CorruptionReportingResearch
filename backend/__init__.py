"""
Backend Package - Corruption Reporting System

This package contains the complete backend implementation of the anonymous
corruption reporting system with AI-powered credibility assessment.

Core Components:
- FastAPI application (main.py)
- Configuration management (config.py)
- Six-layer processing architecture (core/)
- ML model management (models/)
- Supporting services (services/)
- Utilities (utils/)
- Background workers (workers/)

Usage:
    from backend import create_app, get_config
    from backend.constants import APP_NAME, APP_VERSION
    from backend.exceptions import ValidationError
"""

__version__ = "1.0.0"
__author__ = "Research Team"
__description__ = "Anonymous Evidence Validation System"

# Package metadata
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__description__",
    
    # Main application
    "create_app",
    "app",
    
    # Configuration
    "load_config",
    "get_config",
    "AppConfig",
    
    # Logging
    "setup_logging",
    "get_logger",
    
    # Constants
    "constants",
    
    # Exceptions
    "exceptions",
    "BaseApplicationException",
    "ValidationError",
    "NotFoundError",
    "RateLimitError",
    
    # Core layers (for testing/debugging)
    "AnonymityLayer",
    "CredibilityAssessmentLayer",
    "CoordinationDetectionLayer",
    "ConsensusLayer",
    "CounterEvidenceLayer",
    "ReportingLayer",
    "ProcessingOrchestrator",
    
    # Services
    "StorageService",
    "HashChainService",
    "CryptoService",
    "ValidationService",
    "RateLimiterService",
    "MetricsService",
    
    # Models
    "get_clip_model",
    "get_wav2vec_model",
    "get_blip_model",
    "get_sentence_transformer",
]


# ==================== IMPORTS ====================

# Core application
from backend.main import create_app, app

# Configuration
from backend.config import (
    load_config,
    get_config,
    AppConfig
)

# Logging
from backend.logging_config import (
    setup_logging,
    get_logger
)

# Constants (import module, not individual constants to avoid bloat)
from backend import constants

# Exceptions (import module and commonly used exceptions)
from backend import exceptions
from backend.exceptions import (
    BaseApplicationException,
    ValidationError,
    NotFoundError,
    RateLimitError
)

# Core layers
from backend.core.layer1_anonymity import AnonymityLayer
from backend.core.layer2_credibility import CredibilityAssessmentLayer
from backend.core.layer3_coordination import CoordinationDetectionLayer
from backend.core.layer4_consensus import ConsensusLayer
from backend.core.layer5_counter_evidence import CounterEvidenceLayer
from backend.core.layer6_reporting import ReportingLayer
from backend.core.orchestrator import ProcessingOrchestrator

# Services
from backend.services.storage_service import StorageService
from backend.services.hash_chain_service import HashChainService
from backend.services.crypto_service import CryptoService
from backend.services.validation_service import ValidationService
from backend.services.rate_limiter import RateLimiterService
from backend.services.metrics_service import MetricsService

# Models
from backend.models import (
    get_clip_model,
    get_wav2vec_model,
    get_blip_model,
    get_sentence_transformer
)


# ==================== PACKAGE INITIALIZATION ====================

def get_package_info():
    """
    Get package information.
    
    Returns:
        dict: Package metadata
    """
    return {
        "name": "backend",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "components": {
            "api": "FastAPI REST endpoints",
            "core": "Six-layer processing architecture",
            "models": "ML model management",
            "services": "Supporting services",
            "utils": "Utility functions",
            "workers": "Background processing"
        }
    }


def verify_installation():
    """
    Verify package installation and dependencies.
    
    Returns:
        dict: Installation status
    """
    status = {
        "installed": True,
        "missing_dependencies": [],
        "warnings": []
    }
    
    # Check critical dependencies
    required_packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("networkx", "NetworkX"),
        ("sklearn", "scikit-learn"),
        ("reportlab", "ReportLab"),
        ("cryptography", "Cryptography"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow")
    ]
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            status["installed"] = False
            status["missing_dependencies"].append(package_name)
    
    # Check optional dependencies
    optional_packages = [
        ("scipy", "SciPy"),
    ]
    
    for import_name, package_name in optional_packages:
        try:
            __import__(import_name)
        except ImportError:
            status["warnings"].append(f"Optional package {package_name} not installed")
    
    return status


# ==================== PACKAGE-LEVEL LOGGING ====================

# Initialize package logger
_logger = get_logger(__name__)
_logger.debug(f"Backend package initialized (version {__version__})")

# Verify installation on import (only log warnings)
_install_status = verify_installation()
if not _install_status["installed"]:
    _logger.error(
        f"Missing required dependencies: {', '.join(_install_status['missing_dependencies'])}"
    )
    _logger.error("Install with: pip install -r requirements.txt")

if _install_status["warnings"]:
    for warning in _install_status["warnings"]:
        _logger.debug(warning)


# ==================== CONVENIENCE FUNCTIONS ====================

def quick_start(
    host: str = "0.0.0.0",
    port: int = 8000,
    config_file: str = None,
    environment: str = None
):
    """
    Quick start the application with minimal configuration.
    
    Args:
        host: Server host
        port: Server port
        config_file: Configuration file path
        environment: Environment name
    
    Example:
        >>> from backend import quick_start
        >>> quick_start(port=8000)
    """
    import uvicorn
    
    # Load configuration
    config = load_config(config_file, environment)
    
    # Override with function parameters
    config.server.host = host
    config.server.port = port
    
    # Create and run app
    _logger.info(f"Quick starting server at http://{host}:{port}")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=config.server.reload
    )


def initialize_storage():
    """
    Initialize storage directories and files.
    
    Example:
        >>> from backend import initialize_storage
        >>> initialize_storage()
    """
    _logger.info("Initializing storage...")
    
    storage = StorageService()
    storage.initialize()
    
    _logger.info("Storage initialized successfully")


def reset_storage(confirm: bool = False):
    """
    Reset storage (delete all data).
    
    WARNING: This will delete all submissions, evidence, and reports!
    
    Args:
        confirm: Must be True to proceed
    
    Example:
        >>> from backend import reset_storage
        >>> reset_storage(confirm=True)
    """
    if not confirm:
        raise ValueError(
            "Storage reset requires explicit confirmation. "
            "Call with confirm=True to proceed."
        )
    
    _logger.warning("Resetting storage (deleting all data)...")
    
    from backend.constants import DATA_DIR
    import shutil
    
    # Remove data directory
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    
    # Reinitialize
    initialize_storage()
    
    _logger.info("Storage reset complete")


def get_system_info():
    """
    Get system information for debugging.
    
    Returns:
        dict: System information
    """
    import sys
    import platform
    
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
    except ImportError:
        torch_version = "Not installed"
        cuda_available = False
    
    return {
        "package": get_package_info(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "platform": platform.platform()
        },
        "pytorch": {
            "version": torch_version,
            "cuda_available": cuda_available
        },
        "installation": verify_installation()
    }


# ==================== CLI HELPER ====================

def main():
    """
    CLI entry point for package.
    
    Usage:
        python -m backend
        python -m backend --help
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description=f"{__description__} v{__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run server
    run_parser = subparsers.add_parser("run", help="Run the server")
    run_parser.add_argument("--host", default="0.0.0.0", help="Host address")
    run_parser.add_argument("--port", type=int, default=8000, help="Port number")
    run_parser.add_argument("--config", help="Config file path")
    run_parser.add_argument("--env", help="Environment name")
    
    # Initialize storage
    subparsers.add_parser("init", help="Initialize storage")
    
    # System info
    subparsers.add_parser("info", help="Show system information")
    
    # Verify installation
    subparsers.add_parser("verify", help="Verify installation")
    
    args = parser.parse_args()
    
    if args.command == "run":
        quick_start(
            host=args.host,
            port=args.port,
            config_file=args.config,
            environment=args.env
        )
    
    elif args.command == "init":
        initialize_storage()
        print("Storage initialized successfully")
    
    elif args.command == "info":
        import json
        info = get_system_info()
        print(json.dumps(info, indent=2))
    
    elif args.command == "verify":
        status = verify_installation()
        
        if status["installed"]:
            print("✓ All required dependencies installed")
        else:
            print("✗ Missing dependencies:")
            for dep in status["missing_dependencies"]:
                print(f"  - {dep}")
            print("\nInstall with: pip install -r requirements.txt")
            sys.exit(1)
        
        if status["warnings"]:
            print("\nWarnings:")
            for warning in status["warnings"]:
                print(f"  - {warning}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
