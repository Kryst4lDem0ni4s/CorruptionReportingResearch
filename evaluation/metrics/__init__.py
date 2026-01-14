"""
Corruption Reporting System - Metrics Package
Version: 1.0.0
Description: Performance metrics computation for evaluation experiments

This package provides:
- Base metric interface (BaseMetric)
- Deepfake detection metrics (AUROC, precision, recall)
- Coordination detection metrics (graph analysis)
- Consensus simulation metrics (convergence time)
- Counter-evidence metrics (Bayesian impact)

Usage:
    from evaluation.metrics import BaseMetric, MetricRegistry
    from evaluation.metrics import compute_auroc, compute_precision_recall
    
    # Use specific metrics
    auroc = compute_auroc(y_true, y_pred)
    
    # Register custom metric
    registry = MetricRegistry()
    registry.register('custom_metric', CustomMetric())
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union
import logging
import numpy as np

# Package root
PACKAGE_ROOT = Path(__file__).parent.resolve()

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.metrics')

# ============================================
# IMPORTS
# ============================================

# Import base metric
try:
    from evaluation.metrics.base_metric import (
        BaseMetric,
        MetricResult,
        MetricRegistry
    )
except ImportError as e:
    logger.warning(f"Could not import base_metric: {e}")
    BaseMetric = None
    MetricResult = None
    MetricRegistry = None

# ============================================
# METRIC CATEGORIES
# ============================================

METRIC_CATEGORIES = {
    'deepfake_detection': {
        'description': 'Deepfake detection performance metrics',
        'metrics': ['auroc', 'precision', 'recall', 'f1_score', 'accuracy'],
        'target': {'auroc': 0.75}  # MVP target (paper: 0.90)
    },
    'coordination_detection': {
        'description': 'Coordination attack detection metrics',
        'metrics': ['precision', 'recall', 'f1_score', 'detection_rate'],
        'target': {'precision': 0.70, 'recall': 0.70}
    },
    'consensus': {
        'description': 'Byzantine consensus simulation metrics',
        'metrics': ['convergence_time', 'agreement_rate', 'fault_tolerance'],
        'target': {'agreement_rate': 0.80}
    },
    'counter_evidence': {
        'description': 'Counter-evidence Bayesian aggregation metrics',
        'metrics': ['false_positive_reduction', 'decision_change_rate'],
        'target': {'false_positive_reduction': 0.20}  # 20%+ reduction
    },
    'performance': {
        'description': 'System performance benchmarks',
        'metrics': ['latency_ms', 'throughput', 'memory_mb'],
        'target': {'latency_ms': 5000, 'throughput': 20, 'memory_mb': 8192}
    }
}

# ============================================
# UTILITY FUNCTIONS
# ============================================

def validate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray] = None
) -> bool:
    """
    Validate prediction arrays
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_score: Prediction scores (optional)
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If validation fails
    """
    # Check types
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
    
    # Check shapes
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    
    # Check scores if provided
    if y_score is not None:
        if not isinstance(y_score, np.ndarray):
            y_score = np.array(y_score)
        if y_score.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Shape mismatch: y_true {y_true.shape} vs y_score {y_score.shape}"
            )
    
    # Check for empty arrays
    if len(y_true) == 0:
        raise ValueError("Empty prediction arrays")
    
    return True

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Compute confusion matrix components
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        
    Returns:
        Dictionary with TP, TN, FP, FN counts
    """
    validate_predictions(y_true, y_pred)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute confusion matrix
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    return {
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'total': int(len(y_true))
    }

def compute_basic_metrics(
    confusion_matrix: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute basic metrics from confusion matrix
    
    Args:
        confusion_matrix: Dictionary with TP, TN, FP, FN
        
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    tp = confusion_matrix['true_positives']
    tn = confusion_matrix['true_negatives']
    fp = confusion_matrix['false_positives']
    fn = confusion_matrix['false_negatives']
    
    # Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall (sensitivity, true positive rate)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Specificity (true negative rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'specificity': round(specificity, 4)
    }

def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Compute confidence interval using bootstrap
    
    Args:
        data: Data array
        confidence: Confidence level (default: 0.95)
        
    Returns:
        Dictionary with mean, lower, upper bounds
    """
    if len(data) == 0:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}
    
    mean = np.mean(data)
    
    # Use percentile method for confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    
    return {
        'mean': round(float(mean), 4),
        'lower': round(float(lower), 4),
        'upper': round(float(upper), 4),
        'confidence': confidence
    }

def compare_metrics(
    baseline: Dict[str, float],
    experimental: Dict[str, float],
    metric_names: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare baseline vs experimental metrics
    
    Args:
        baseline: Baseline metrics
        experimental: Experimental metrics
        metric_names: Specific metrics to compare (None = all)
        
    Returns:
        Comparison dictionary with improvements
    """
    if metric_names is None:
        metric_names = list(set(baseline.keys()) & set(experimental.keys()))
    
    comparison = {}
    
    for metric_name in metric_names:
        if metric_name not in baseline or metric_name not in experimental:
            continue
        
        baseline_value = baseline[metric_name]
        experimental_value = experimental[metric_name]
        
        # Compute improvement
        if baseline_value != 0:
            improvement = (experimental_value - baseline_value) / baseline_value
        else:
            improvement = 1.0 if experimental_value > 0 else 0.0
        
        comparison[metric_name] = {
            'baseline': baseline_value,
            'experimental': experimental_value,
            'absolute_improvement': experimental_value - baseline_value,
            'relative_improvement': improvement,
            'improved': experimental_value > baseline_value
        }
    
    return comparison

# ============================================
# METRIC REGISTRY
# ============================================

# Global registry instance
_global_registry = None

def get_registry() -> 'MetricRegistry':
    """Get global metric registry"""
    global _global_registry
    if _global_registry is None and MetricRegistry is not None:
        _global_registry = MetricRegistry()
    return _global_registry

def register_metric(name: str, metric: 'BaseMetric'):
    """Register a metric in global registry"""
    registry = get_registry()
    if registry:
        registry.register(name, metric)

def get_metric(name: str) -> Optional['BaseMetric']:
    """Get metric from global registry"""
    registry = get_registry()
    if registry:
        return registry.get(name)
    return None

# ============================================
# METRIC INFORMATION
# ============================================

def list_metrics() -> List[str]:
    """List all available metrics"""
    all_metrics = []
    for category_info in METRIC_CATEGORIES.values():
        all_metrics.extend(category_info['metrics'])
    return all_metrics

def get_metric_info(metric_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a specific metric"""
    for category_name, category_info in METRIC_CATEGORIES.items():
        if metric_name in category_info['metrics']:
            return {
                'name': metric_name,
                'category': category_name,
                'description': category_info['description'],
                'target': category_info['target'].get(metric_name)
            }
    return None

def get_category_info(category_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a metric category"""
    return METRIC_CATEGORIES.get(category_name)

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    # Base classes
    'BaseMetric',
    'MetricResult',
    'MetricRegistry',
    
    # Registry functions
    'get_registry',
    'register_metric',
    'get_metric',
    
    # Utility functions
    'validate_predictions',
    'compute_confusion_matrix',
    'compute_basic_metrics',
    'compute_confidence_interval',
    'compare_metrics',
    
    # Information functions
    'list_metrics',
    'get_metric_info',
    'get_category_info',
    
    # Constants
    'METRIC_CATEGORIES',
    'PACKAGE_ROOT'
]

# Version
__version__ = '1.0.0'

# Log package initialization
logger.info(f"Metrics package initialized: {PACKAGE_ROOT}")
logger.info(f"Available metric categories: {', '.join(METRIC_CATEGORIES.keys())}")
