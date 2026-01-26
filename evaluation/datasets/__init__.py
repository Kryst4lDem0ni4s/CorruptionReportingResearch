"""
Corruption Reporting System - Datasets Package
Version: 1.0.0
Description: Dataset management for evaluation experiments

This package provides:
- Dataset downloading and caching
- Dataset loading and preprocessing
- Synthetic data generation
- Data validation

Supported Datasets:
- FaceForensics++ (deepfake detection)
- Celeb-DF v2 (deepfake detection)
- Synthetic coordinated attacks (coordination detection)

Usage:
    from evaluation.datasets import download_dataset, load_dataset
    
    # Download dataset
    download_dataset('faceforensics', sample_size=100)
    
    # Load dataset
    dataset = load_dataset('faceforensics', split='test')
    
    # Generate synthetic attacks
    synthetic_data = generate_synthetic_attacks(num_groups=10)
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging

# Package root
PACKAGE_ROOT = Path(__file__).parent.resolve()
DATASETS_DIR = PACKAGE_ROOT
CACHE_DIR = DATASETS_DIR / '.cache'
CACHE_DIR.mkdir(exist_ok=True)

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.datasets')

# ============================================
# DATASET REGISTRY
# ============================================

DATASET_REGISTRY = {
    'faceforensics': {
        'name': 'FaceForensics++',
        'type': 'deepfake_detection',
        'url': 'https://github.com/ondyari/FaceForensics',
        'size_gb': 500,  # Full dataset
        'num_samples': 1000,  # Original video count
        'formats': ['images', 'videos'],
        'license': 'Research use only'
    },
    'celebdf': {
        'name': 'Celeb-DF v2',
        'type': 'deepfake_detection',
        'url': 'https://github.com/yuezunli/celeb-deepfakeforensics',
        'size_gb': 50,
        'num_samples': 590,  # Original video count
        'formats': ['videos'],
        'license': 'Research use only'
    },
    'real-and-fake-face-detection': {
        'name': 'Real and Fake Face Detection',
        'type': 'deepfake_detection',
        'url': 'https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection',
        'size_gb': 1.0,
        'num_samples': 2041,
        'formats': ['images'],
        'license': 'Research use only'
    },
    'synthetic_attacks': {
        'name': 'Synthetic Coordinated Attacks',
        'type': 'coordination_detection',
        'generated': True,
        'configurable': True
    }
}

# ============================================
# DATASET PATHS
# ============================================

def get_dataset_path(dataset_name: str) -> Path:
    """
    Get path to dataset directory
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Path to dataset directory
    """
    dataset_dir = DATASETS_DIR / dataset_name
    dataset_dir.mkdir(exist_ok=True)
    return dataset_dir

def get_cache_path(dataset_name: str, filename: str) -> Path:
    """
    Get path to cached file
    
    Args:
        dataset_name: Name of dataset
        filename: Cache filename
        
    Returns:
        Path to cached file
    """
    cache_dir = CACHE_DIR / dataset_name
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / filename

# ============================================
# DATASET INFO
# ============================================

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """
    Get information about a dataset
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Dataset information dictionary
        
    Raises:
        ValueError: If dataset not found
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DATASET_REGISTRY[dataset_name]

def list_datasets() -> List[str]:
    """
    List available datasets
    
    Returns:
        List of dataset names
    """
    return list(DATASET_REGISTRY.keys())

def is_dataset_downloaded(dataset_name: str) -> bool:
    """
    Check if dataset is downloaded
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        True if dataset exists locally
    """
    dataset_path = get_dataset_path(dataset_name)
    
    # Check if directory has files
    if not dataset_path.exists():
        return False
    
    # Check for marker file OR if directory is not empty (manual download support)
    marker_file = dataset_path / '.downloaded'
    if marker_file.exists():
        return True
        
    # Fallback: check if directory has content (recursive check for files)
    has_files = any(dataset_path.rglob('*.*'))
    if has_files:
        logger.info(f"Dataset {dataset_name} found (no .downloaded marker)")
        return True
        
    return False

# ============================================
# DATASET STATISTICS
# ============================================

def get_dataset_statistics(dataset_name: str) -> Dict[str, Any]:
    """
    Get statistics about downloaded dataset
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Statistics dictionary
    """
    dataset_path = get_dataset_path(dataset_name)
    
    if not dataset_path.exists():
        return {
            'exists': False,
            'num_files': 0,
            'size_bytes': 0
        }
    
    # Count files
    num_files = sum(1 for _ in dataset_path.rglob('*') if _.is_file())
    
    # Calculate size
    size_bytes = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
    
    return {
        'exists': True,
        'num_files': num_files,
        'size_bytes': size_bytes,
        'size_mb': size_bytes / (1024 * 1024),
        'path': str(dataset_path)
    }

# ============================================
# VALIDATION
# ============================================

def validate_dataset(dataset_name: str) -> Tuple[bool, List[str]]:
    """
    Validate dataset integrity
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check if dataset exists
    if not is_dataset_downloaded(dataset_name):
        errors.append(f"Dataset {dataset_name} not downloaded")
        return False, errors
    
    dataset_path = get_dataset_path(dataset_name)
    
    # Check if directory is readable
    if not os.access(dataset_path, os.R_OK):
        errors.append(f"Dataset directory not readable: {dataset_path}")
    
    # Check if has files
    stats = get_dataset_statistics(dataset_name)
    if stats['num_files'] == 0:
        errors.append(f"Dataset directory is empty: {dataset_path}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

# ============================================
# CLEANUP
# ============================================

def clear_cache(dataset_name: Optional[str] = None):
    """
    Clear dataset cache
    
    Args:
        dataset_name: Specific dataset to clear (None = all)
    """
    if dataset_name:
        cache_path = CACHE_DIR / dataset_name
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
            logger.info(f"Cleared cache for {dataset_name}")
    else:
        if CACHE_DIR.exists():
            import shutil
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir()
            logger.info("Cleared all dataset cache")

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    # Paths
    'PACKAGE_ROOT',
    'DATASETS_DIR',
    'CACHE_DIR',
    'get_dataset_path',
    'get_cache_path',
    
    # Registry
    'DATASET_REGISTRY',
    'get_dataset_info',
    'list_datasets',
    
    # Status
    'is_dataset_downloaded',
    'get_dataset_statistics',
    
    # Validation
    'validate_dataset',
    
    # Cleanup
    'clear_cache',
]

# Log package initialization
logger.info(f"Datasets package initialized: {DATASETS_DIR}")
logger.info(f"Available datasets: {', '.join(list_datasets())}")
