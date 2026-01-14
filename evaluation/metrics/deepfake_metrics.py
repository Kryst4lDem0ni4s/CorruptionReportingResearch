"""
Corruption Reporting System - Deepfake Detection Metrics
Version: 1.0.0
Description: Metrics for evaluating deepfake detection performance

This module provides:
- AUROC (Area Under ROC Curve)
- Precision, Recall, F1-Score
- Accuracy metrics
- Confidence intervals
- Per-threshold analysis

Usage:
    from evaluation.metrics.deepfake_metrics import (
        AUROCMetric, PrecisionRecallMetric, DeepfakeMetrics
    )
    
    # Compute AUROC
    auroc_metric = AUROCMetric()
    result = auroc_metric.compute(y_true, y_score)
    print(f"AUROC: {result.value:.4f}")
    
    # Compute all deepfake metrics
    metrics = DeepfakeMetrics()
    results = metrics.compute_all(y_true, y_pred, y_score)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import base metric
try:
    from evaluation.metrics.base_metric import BaseMetric, MetricResult
    from evaluation.metrics import (
        validate_predictions,
        compute_confusion_matrix,
        compute_basic_metrics,
        compute_confidence_interval
    )
except ImportError as e:
    print(f"Error importing base metrics: {e}")
    sys.exit(1)

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.metrics.deepfake')

# ============================================
# AUROC METRIC
# ============================================

class AUROCMetric(BaseMetric):
    """
    Area Under ROC Curve (AUROC) metric
    
    Measures the ability of the classifier to distinguish between
    real and fake samples across all classification thresholds.
    
    Target: ≥0.75 (MVP), ≥0.90 (paper goal)
    """
    
    def __init__(self):
        """Initialize AUROC metric"""
        super().__init__(
            name='auroc',
            description='Area Under ROC Curve',
            category='deepfake_detection'
        )
    
    def validate_inputs(self, y_true: np.ndarray, y_score: np.ndarray, **kwargs) -> bool:
        """
        Validate inputs for AUROC computation
        
        Args:
            y_true: Ground truth labels (0/1)
            y_score: Prediction scores (0-1)
            
        Returns:
            True if valid
        """
        validate_predictions(y_true, None, y_score)
        
        # Check that we have both classes
        unique_labels = np.unique(y_true)
        if len(unique_labels) < 2:
            raise ValueError(
                f"AUROC requires both classes, got only: {unique_labels}"
            )
        
        return True
    
    def compute(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        **kwargs
    ) -> MetricResult:
        """
        Compute AUROC
        
        Args:
            y_true: Ground truth labels (0/1)
            y_score: Prediction scores (0-1)
            
        Returns:
            MetricResult with AUROC value
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # Compute ROC curve points
        fpr, tpr, thresholds = self._compute_roc_curve(y_true, y_score)
        
        # Compute AUROC using trapezoidal rule
        auroc = self._compute_auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J statistic)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        return MetricResult(
            name=self.name,
            value=round(float(auroc), 4),
            metadata={
                'target_mvp': 0.75,
                'target_paper': 0.90,
                'meets_mvp_target': auroc >= 0.75,
                'meets_paper_target': auroc >= 0.90,
                'optimal_threshold': round(float(optimal_threshold), 4),
                'optimal_tpr': round(float(tpr[optimal_idx]), 4),
                'optimal_fpr': round(float(fpr[optimal_idx]), 4),
                'num_samples': len(y_true),
                'num_thresholds': len(thresholds)
            },
            sub_metrics={
                'optimal_threshold': round(float(optimal_threshold), 4),
                'optimal_sensitivity': round(float(tpr[optimal_idx]), 4),
                'optimal_specificity': round(float(1 - fpr[optimal_idx]), 4)
            }
        )
    
    def _compute_roc_curve(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute ROC curve
        
        Args:
            y_true: Ground truth labels
            y_score: Prediction scores
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        # Get unique thresholds (sorted in descending order)
        thresholds = np.unique(y_score)[::-1]
        
        # Add boundary thresholds
        thresholds = np.concatenate(([thresholds[0] + 1], thresholds, [thresholds[-1] - 1]))
        
        # Compute TPR and FPR at each threshold
        tpr_list = []
        fpr_list = []
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            
            # Compute confusion matrix
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            
            # Compute TPR and FPR
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        return np.array(fpr_list), np.array(tpr_list), thresholds
    
    def _compute_auc(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute area under curve using trapezoidal rule
        
        Args:
            x: X coordinates (FPR)
            y: Y coordinates (TPR)
            
        Returns:
            Area under curve
        """
        # Sort by x
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        
        # Compute area using trapezoidal rule
        area = np.trapz(y, x)
        
        return area
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['y_true', 'y_score'],
            'optional': [],
            'output_type': 'float'
        }
    
    def supports_confidence_interval(self) -> bool:
        """AUROC supports bootstrap confidence intervals"""
        return True

# ============================================
# PRECISION-RECALL METRIC
# ============================================

class PrecisionRecallMetric(BaseMetric):
    """
    Precision, Recall, and F1-Score metrics
    
    Measures classification performance at a specific threshold.
    """
    
    def __init__(self):
        """Initialize precision-recall metric"""
        super().__init__(
            name='precision_recall',
            description='Precision, Recall, F1-Score',
            category='deepfake_detection'
        )
    
    def validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        validate_predictions(y_true, y_pred)
        return True
    
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> MetricResult:
        """
        Compute precision, recall, F1
        
        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Predicted labels (0/1)
            
        Returns:
            MetricResult with metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred)
        
        # Compute basic metrics
        metrics = compute_basic_metrics(cm)
        
        return MetricResult(
            name=self.name,
            value=metrics['f1_score'],  # Primary value is F1
            metadata={
                'confusion_matrix': cm,
                'num_samples': cm['total']
            },
            sub_metrics={
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy'],
                'specificity': metrics['specificity']
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['y_true', 'y_pred'],
            'optional': [],
            'output_type': 'dict'
        }

# ============================================
# ACCURACY METRIC
# ============================================

class AccuracyMetric(BaseMetric):
    """
    Classification accuracy metric
    
    Measures the proportion of correct predictions.
    """
    
    def __init__(self):
        """Initialize accuracy metric"""
        super().__init__(
            name='accuracy',
            description='Classification Accuracy',
            category='deepfake_detection'
        )
    
    def validate_inputs(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        validate_predictions(y_true, y_pred)
        return True
    
    def compute(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **kwargs
    ) -> MetricResult:
        """
        Compute accuracy
        
        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Predicted labels (0/1)
            
        Returns:
            MetricResult with accuracy
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute accuracy
        correct = np.sum(y_true == y_pred)
        total = len(y_true)
        accuracy = correct / total if total > 0 else 0.0
        
        # Compute per-class accuracy
        real_mask = y_true == 0
        fake_mask = y_true == 1
        
        real_accuracy = np.sum((y_true == y_pred) & real_mask) / np.sum(real_mask) if np.sum(real_mask) > 0 else 0
        fake_accuracy = np.sum((y_true == y_pred) & fake_mask) / np.sum(fake_mask) if np.sum(fake_mask) > 0 else 0
        
        return MetricResult(
            name=self.name,
            value=round(float(accuracy), 4),
            metadata={
                'num_correct': int(correct),
                'num_total': int(total),
                'num_real': int(np.sum(real_mask)),
                'num_fake': int(np.sum(fake_mask))
            },
            sub_metrics={
                'real_accuracy': round(float(real_accuracy), 4),
                'fake_accuracy': round(float(fake_accuracy), 4),
                'balanced_accuracy': round(float((real_accuracy + fake_accuracy) / 2), 4)
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['y_true', 'y_pred'],
            'optional': [],
            'output_type': 'float'
        }

# ============================================
# THRESHOLD ANALYSIS METRIC
# ============================================

class ThresholdAnalysisMetric(BaseMetric):
    """
    Analyze performance across multiple thresholds
    
    Helps select optimal classification threshold.
    """
    
    def __init__(self):
        """Initialize threshold analysis metric"""
        super().__init__(
            name='threshold_analysis',
            description='Performance across thresholds',
            category='deepfake_detection'
        )
    
    def validate_inputs(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        validate_predictions(y_true, None, y_score)
        return True
    
    def compute(
        self,
        y_true: np.ndarray,
        y_score: np.ndarray,
        thresholds: Optional[List[float]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute metrics at different thresholds
        
        Args:
            y_true: Ground truth labels (0/1)
            y_score: Prediction scores (0-1)
            thresholds: List of thresholds to evaluate
            
        Returns:
            MetricResult with threshold analysis
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        # Default thresholds
        if thresholds is None:
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        
        # Analyze each threshold
        results = []
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            
            # Compute metrics
            cm = compute_confusion_matrix(y_true, y_pred)
            metrics = compute_basic_metrics(cm)
            
            result = {
                'threshold': threshold,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'accuracy': metrics['accuracy']
            }
            results.append(result)
            
            # Track best F1
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                best_threshold = threshold
        
        return MetricResult(
            name=self.name,
            value=best_threshold,
            metadata={
                'best_f1': best_f1,
                'num_thresholds': len(thresholds),
                'threshold_range': [min(thresholds), max(thresholds)]
            },
            sub_metrics={
                'best_threshold': best_threshold,
                'best_f1': best_f1,
                'analysis': results
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['y_true', 'y_score'],
            'optional': ['thresholds'],
            'output_type': 'float'
        }

# ============================================
# COMBINED DEEPFAKE METRICS
# ============================================

class DeepfakeMetrics:
    """
    Compute all deepfake detection metrics
    
    Convenience class for computing all metrics at once.
    """
    
    def __init__(self):
        """Initialize all metrics"""
        self.auroc_metric = AUROCMetric()
        self.pr_metric = PrecisionRecallMetric()
        self.accuracy_metric = AccuracyMetric()
        self.threshold_metric = ThresholdAnalysisMetric()
        
        logger.info("Initialized DeepfakeMetrics")
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Compute all deepfake metrics
        
        Args:
            y_true: Ground truth labels (0/1)
            y_pred: Predicted labels (0/1)
            y_score: Prediction scores (0-1, optional)
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Compute precision, recall, F1
        try:
            results['precision_recall'] = self.pr_metric.compute(y_true, y_pred)
        except Exception as e:
            logger.error(f"Failed to compute precision/recall: {e}")
        
        # Compute accuracy
        try:
            results['accuracy'] = self.accuracy_metric.compute(y_true, y_pred)
        except Exception as e:
            logger.error(f"Failed to compute accuracy: {e}")
        
        # Compute AUROC if scores provided
        if y_score is not None:
            try:
                results['auroc'] = self.auroc_metric.compute(y_true, y_score)
            except Exception as e:
                logger.error(f"Failed to compute AUROC: {e}")
            
            # Compute threshold analysis
            try:
                results['threshold_analysis'] = self.threshold_metric.compute(
                    y_true, y_score
                )
            except Exception as e:
                logger.error(f"Failed to compute threshold analysis: {e}")
        
        logger.info(f"Computed {len(results)} metric(s)")
        
        return results
    
    def get_summary(
        self,
        results: Dict[str, MetricResult]
    ) -> Dict[str, Any]:
        """
        Get summary of results
        
        Args:
            results: Dictionary of metric results
            
        Returns:
            Summary dictionary
        """
        summary = {}
        
        # Extract key metrics
        if 'auroc' in results:
            auroc_result = results['auroc']
            summary['auroc'] = auroc_result.value
            summary['meets_mvp_target'] = auroc_result.metadata.get('meets_mvp_target', False)
            summary['optimal_threshold'] = auroc_result.sub_metrics.get('optimal_threshold')
        
        if 'precision_recall' in results:
            pr_result = results['precision_recall']
            summary['precision'] = pr_result.sub_metrics.get('precision')
            summary['recall'] = pr_result.sub_metrics.get('recall')
            summary['f1_score'] = pr_result.sub_metrics.get('f1_score')
        
        if 'accuracy' in results:
            acc_result = results['accuracy']
            summary['accuracy'] = acc_result.value
            summary['balanced_accuracy'] = acc_result.sub_metrics.get('balanced_accuracy')
        
        return summary

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute AUROC
    
    Args:
        y_true: Ground truth labels
        y_score: Prediction scores
        
    Returns:
        AUROC value
    """
    metric = AUROCMetric()
    result = metric.compute(y_true, y_score)
    return result.value

def compute_precision_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute precision, recall, F1
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    metric = PrecisionRecallMetric()
    result = metric.compute(y_true, y_pred)
    return result.sub_metrics

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'AUROCMetric',
    'PrecisionRecallMetric',
    'AccuracyMetric',
    'ThresholdAnalysisMetric',
    'DeepfakeMetrics',
    'compute_auroc',
    'compute_precision_recall'
]
