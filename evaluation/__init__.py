"""
Corruption Reporting System - Evaluation Package
Version: 1.0.0
Description: Research evaluation suite for academic publication

This package provides:
- Dataset loading and preprocessing
- Metrics computation (AUROC, precision, recall)
- Visualization generation
- Benchmark testing
- Performance profiling

Module Structure:
- datasets/: Dataset downloaders and loaders
- metrics/: Performance metric calculators
- visualizations/: Research figure generators
- benchmarks/: Performance testing tools
- results/: Output storage

Usage:
    from evaluation import run_evaluation, load_dataset, compute_metrics
    
    # Run full evaluation
    results = run_evaluation(config_path='config_evaluation.yaml')
    
    # Load specific dataset
    dataset = load_dataset('faceforensics', split='test')
    
    # Compute metrics
    metrics = compute_metrics(predictions, ground_truth, metric_type='deepfake')
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings

# ============================================
# PACKAGE METADATA
# ============================================

__version__ = '1.0.0'
__author__ = 'Corruption Reporting Research Team'
__email__ = 'research@corruption-reporting.org'
__description__ = 'Evaluation suite for corruption reporting system research'

# ============================================
# PACKAGE CONFIGURATION
# ============================================

# Package root directory
PACKAGE_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = PACKAGE_ROOT.parent.resolve()

# Results directory
RESULTS_DIR = PACKAGE_ROOT / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Datasets directory
DATASETS_DIR = PACKAGE_ROOT / 'datasets'

# Figures directory
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

# ============================================
# LOGGING CONFIGURATION
# ============================================

def setup_evaluation_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Setup logging for evaluation package
    
    Args:
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger('evaluation')
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(console_format)
        logger.addHandler(file_handler)
    
    return logger

# Initialize default logger
logger = setup_evaluation_logging(
    level=logging.INFO,
    log_file=RESULTS_DIR / 'evaluation.log'
)

# ============================================
# SUPPRESS WARNINGS
# ============================================

# Suppress common research warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='torch')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ============================================
# ENVIRONMENT VALIDATION
# ============================================

def validate_environment() -> Dict[str, bool]:
    """
    Validate evaluation environment
    
    Returns:
        Dictionary of validation checks
    """
    checks = {}
    
    # Check Python version
    checks['python_version'] = sys.version_info >= (3, 8)
    
    # Check required packages
    required_packages = [
        'numpy',
        'scipy',
        'matplotlib',
        'sklearn',
        'torch',
        'transformers'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            checks[f'package_{package}'] = True
        except ImportError:
            checks[f'package_{package}'] = False
            logger.warning(f'Package {package} not found')
    
    # Check directories
    checks['results_dir'] = RESULTS_DIR.exists()
    checks['datasets_dir'] = DATASETS_DIR.exists()
    checks['figures_dir'] = FIGURES_DIR.exists()
    
    return checks

# ============================================
# CONFIGURATION LOADING
# ============================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load evaluation configuration
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = PACKAGE_ROOT / 'config_evaluation.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f'Config file not found: {config_path}')
        return get_default_config()
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f'Loaded configuration from {config_path}')
        return config
    except ImportError:
        logger.warning('PyYAML not installed, using default config')
        return get_default_config()
    except Exception as e:
        logger.error(f'Failed to load config: {e}')
        return get_default_config()

def get_default_config() -> Dict[str, Any]:
    """
    Get default evaluation configuration
    
    Returns:
        Default configuration dictionary
    """
    return {
        'datasets': {
            'faceforensics': {
                'enabled': True,
                'sample_size': 100,
                'split': 'test'
            },
            'celebdf': {
                'enabled': False,
                'sample_size': 50,
                'split': 'test'
            },
            'synthetic': {
                'enabled': True,
                'num_attacks': 10,
                'attack_size': 5
            }
        },
        'metrics': {
            'deepfake_detection': {
                'auroc_threshold': 0.75,
                'precision_threshold': 0.70,
                'recall_threshold': 0.70
            },
            'coordination_detection': {
                'precision_threshold': 0.65,
                'recall_threshold': 0.65
            },
            'consensus': {
                'convergence_threshold': 0.80,
                'max_iterations': 10
            }
        },
        'visualization': {
            'dpi': 300,
            'format': 'png',
            'style': 'seaborn'
        },
        'output': {
            'save_predictions': True,
            'save_figures': True,
            'save_metrics': True
        }
    }

# ============================================
# UTILITY FUNCTIONS
# ============================================

def get_results_path(filename: str) -> Path:
    """
    Get path for results file
    
    Args:
        filename: Result filename
        
    Returns:
        Full path to results file
    """
    return RESULTS_DIR / filename

def get_figures_path(filename: str) -> Path:
    """
    Get path for figure file
    
    Args:
        filename: Figure filename
        
    Returns:
        Full path to figure file
    """
    return FIGURES_DIR / filename

def get_dataset_path(dataset_name: str) -> Path:
    """
    Get path for dataset
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Full path to dataset directory
    """
    return DATASETS_DIR / dataset_name

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    # Metadata
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    
    # Directories
    'PACKAGE_ROOT',
    'PROJECT_ROOT',
    'RESULTS_DIR',
    'DATASETS_DIR',
    'FIGURES_DIR',
    
    # Logging
    'logger',
    'setup_evaluation_logging',
    
    # Configuration
    'load_config',
    'get_default_config',
    
    # Utilities
    'validate_environment',
    'get_results_path',
    'get_figures_path',
    'get_dataset_path',
]

# ============================================
# PACKAGE INITIALIZATION
# ============================================

def _initialize_package():
    """Initialize evaluation package"""
    logger.info(f'Initializing evaluation package v{__version__}')
    
    # Validate environment
    checks = validate_environment()
    failed_checks = [k for k, v in checks.items() if not v]
    
    if failed_checks:
        logger.warning(f'Failed environment checks: {failed_checks}')
    else:
        logger.info('Environment validation passed')
    
    # Create required directories
    for directory in [RESULTS_DIR, FIGURES_DIR]:
        directory.mkdir(exist_ok=True)
    
    logger.info('Package initialization complete')

# Run initialization
_initialize_package()
