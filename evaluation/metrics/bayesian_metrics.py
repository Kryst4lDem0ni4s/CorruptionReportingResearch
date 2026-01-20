"""
Corruption Reporting System - Bayesian Counter-Evidence Metrics
Version: 1.0.0
Description: Metrics for evaluating counter-evidence impact

This module provides:
- False positive reduction analysis
- Decision change rate metrics
- Bayesian aggregation accuracy
- Presumption-of-innocence effectiveness (1.3x weighting)
- Identity verification bonus impact (1.2x)

Usage:
    from evaluation.metrics.bayesian_metrics import (
        FalsePositiveReductionMetric, DecisionChangeMetric, BayesianMetrics
    )
    
    # Compute false positive reduction
    fpr_metric = FalsePositiveReductionMetric()
    result = fpr_metric.compute(before_decisions, after_decisions, ground_truth)
    
    # Compute all Bayesian metrics
    metrics = BayesianMetrics()
    results = metrics.compute_all(before, after, ground_truth)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
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
        compute_basic_metrics
    )
except ImportError as e:
    print(f"Error importing base metrics: {e}")
    sys.exit(1)

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.metrics.bayesian')

# ============================================
# FALSE POSITIVE REDUCTION METRIC
# ============================================

class FalsePositiveReductionMetric(BaseMetric):
    """
    False positive reduction metric
    
    Measures reduction in false positives after counter-evidence.
    
    Target: >=20% false positive reduction
    """
    
    def __init__(self):
        """Initialize false positive reduction metric"""
        super().__init__(
            name='false_positive_reduction',
            description='FP reduction after counter-evidence',
            category='counter_evidence'
        )
    
    def validate_inputs(
        self,
        before_predictions: np.ndarray,
        after_predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        validate_predictions(ground_truth, before_predictions)
        validate_predictions(ground_truth, after_predictions)
        
        if len(before_predictions) != len(after_predictions):
            raise ValueError("Before and after predictions must have same length")
        
        return True
    
    def compute(
        self,
        before_predictions: np.ndarray,
        after_predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> MetricResult:
        """
        Compute false positive reduction
        
        Args:
            before_predictions: Predictions before counter-evidence
            after_predictions: Predictions after counter-evidence
            ground_truth: Ground truth labels
            
        Returns:
            MetricResult with FP reduction
        """
        # Convert to numpy arrays
        before_predictions = np.array(before_predictions)
        after_predictions = np.array(after_predictions)
        ground_truth = np.array(ground_truth)
        
        # Compute confusion matrices
        cm_before = compute_confusion_matrix(ground_truth, before_predictions)
        cm_after = compute_confusion_matrix(ground_truth, after_predictions)
        
        # Extract false positives
        fp_before = cm_before['false_positives']
        fp_after = cm_after['false_positives']
        
        # Compute reduction
        fp_reduction = fp_before - fp_after
        fp_reduction_rate = fp_reduction / fp_before if fp_before > 0 else 0.0
        
        # Check target
        meets_target = fp_reduction_rate >= 0.20
        
        # Compute other metrics for context
        metrics_before = compute_basic_metrics(cm_before)
        metrics_after = compute_basic_metrics(cm_after)
        
        return MetricResult(
            name=self.name,
            value=round(float(fp_reduction_rate), 4),
            metadata={
                'target': 0.20,
                'meets_target': meets_target,
                'fp_before': fp_before,
                'fp_after': fp_after,
                'fp_reduction_absolute': fp_reduction,
                'num_samples': len(ground_truth)
            },
            sub_metrics={
                'fp_reduction_rate': round(float(fp_reduction_rate), 4),
                'precision_before': metrics_before['precision'],
                'precision_after': metrics_after['precision'],
                'precision_improvement': round(
                    metrics_after['precision'] - metrics_before['precision'], 4
                ),
                'accuracy_before': metrics_before['accuracy'],
                'accuracy_after': metrics_after['accuracy']
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['before_predictions', 'after_predictions', 'ground_truth'],
            'optional': [],
            'output_type': 'float'
        }

# ============================================
# DECISION CHANGE RATE METRIC
# ============================================

class DecisionChangeMetric(BaseMetric):
    """
    Decision change rate metric
    
    Measures how often counter-evidence changes the final decision.
    """
    
    def __init__(self):
        """Initialize decision change metric"""
        super().__init__(
            name='decision_change_rate',
            description='Rate of decision changes',
            category='counter_evidence'
        )
    
    def validate_inputs(
        self,
        before_decisions: np.ndarray,
        after_decisions: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        if len(before_decisions) != len(after_decisions):
            raise ValueError("Before and after decisions must have same length")
        return True
    
    def compute(
        self,
        before_decisions: np.ndarray,
        after_decisions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute decision change rate
        
        Args:
            before_decisions: Decisions before counter-evidence
            after_decisions: Decisions after counter-evidence
            ground_truth: Ground truth labels (optional)
            
        Returns:
            MetricResult with decision change rate
        """
        # Convert to numpy arrays
        before_decisions = np.array(before_decisions)
        after_decisions = np.array(after_decisions)
        
        # Compute changes
        changes = (before_decisions != after_decisions)
        num_changes = np.sum(changes)
        total_decisions = len(before_decisions)
        
        change_rate = num_changes / total_decisions if total_decisions > 0 else 0.0
        
        # Analyze change correctness if ground truth provided
        change_analysis = {}
        if ground_truth is not None:
            ground_truth = np.array(ground_truth)
            
            # Incorrect → Correct
            incorrect_to_correct = np.sum(
                changes & (before_decisions != ground_truth) & (after_decisions == ground_truth)
            )
            
            # Correct → Incorrect
            correct_to_incorrect = np.sum(
                changes & (before_decisions == ground_truth) & (after_decisions != ground_truth)
            )
            
            # Incorrect → Incorrect (still wrong)
            incorrect_to_incorrect = np.sum(
                changes & (before_decisions != ground_truth) & (after_decisions != ground_truth)
            )
            
            change_analysis = {
                'incorrect_to_correct': int(incorrect_to_correct),
                'correct_to_incorrect': int(correct_to_incorrect),
                'incorrect_to_incorrect': int(incorrect_to_incorrect),
                'net_improvement': int(incorrect_to_correct - correct_to_incorrect)
            }
        
        return MetricResult(
            name=self.name,
            value=round(float(change_rate), 4),
            metadata={
                'num_changes': int(num_changes),
                'num_unchanged': int(total_decisions - num_changes),
                'total_decisions': int(total_decisions),
                'has_ground_truth': ground_truth is not None
            },
            sub_metrics={
                'change_rate': round(float(change_rate), 4),
                'unchanged_rate': round(float(1 - change_rate), 4),
                **change_analysis
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['before_decisions', 'after_decisions'],
            'optional': ['ground_truth'],
            'output_type': 'float'
        }

# ============================================
# BAYESIAN ACCURACY METRIC
# ============================================

class BayesianAccuracyMetric(BaseMetric):
    """
    Bayesian aggregation accuracy metric
    
    Measures accuracy of Bayesian counter-evidence aggregation.
    """
    
    def __init__(self):
        """Initialize Bayesian accuracy metric"""
        super().__init__(
            name='bayesian_accuracy',
            description='Bayesian aggregation accuracy',
            category='counter_evidence'
        )
    
    def validate_inputs(
        self,
        aggregated_predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        validate_predictions(ground_truth, aggregated_predictions)
        return True
    
    def compute(
        self,
        aggregated_predictions: np.ndarray,
        ground_truth: np.ndarray,
        evidence_weights: Optional[List[float]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute Bayesian accuracy
        
        Args:
            aggregated_predictions: Predictions after Bayesian aggregation
            ground_truth: Ground truth labels
            evidence_weights: Weights used in aggregation (optional)
            
        Returns:
            MetricResult with accuracy
        """
        # Convert to numpy arrays
        aggregated_predictions = np.array(aggregated_predictions)
        ground_truth = np.array(ground_truth)
        
        # Compute confusion matrix and metrics
        cm = compute_confusion_matrix(ground_truth, aggregated_predictions)
        metrics = compute_basic_metrics(cm)
        
        # Analyze weighting effectiveness if provided
        weight_analysis = {}
        if evidence_weights:
            weight_analysis = {
                'mean_weight': round(float(np.mean(evidence_weights)), 4),
                'std_weight': round(float(np.std(evidence_weights)), 4),
                'min_weight': round(float(np.min(evidence_weights)), 4),
                'max_weight': round(float(np.max(evidence_weights)), 4)
            }
        
        return MetricResult(
            name=self.name,
            value=metrics['accuracy'],
            metadata={
                'confusion_matrix': cm,
                'has_weights': bool(evidence_weights)
            },
            sub_metrics={
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                **weight_analysis
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['aggregated_predictions', 'ground_truth'],
            'optional': ['evidence_weights'],
            'output_type': 'float'
        }

# ============================================
# PRESUMPTION WEIGHTING METRIC
# ============================================

class PresumptionWeightingMetric(BaseMetric):
    """
    Presumption-of-innocence weighting effectiveness metric
    
    Measures impact of 1.3× presumption-of-innocence weighting.
    """
    
    def __init__(self):
        """Initialize presumption weighting metric"""
        super().__init__(
            name='presumption_weighting',
            description='Presumption-of-innocence weighting effectiveness',
            category='counter_evidence'
        )
    
    def validate_inputs(
        self,
        without_weighting: np.ndarray,
        with_weighting: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        validate_predictions(ground_truth, without_weighting)
        validate_predictions(ground_truth, with_weighting)
        return True
    
    def compute(
        self,
        without_weighting: np.ndarray,
        with_weighting: np.ndarray,
        ground_truth: np.ndarray,
        weighting_factor: float = 1.3,
        **kwargs
    ) -> MetricResult:
        """
        Compute presumption weighting effectiveness
        
        Args:
            without_weighting: Predictions without presumption weighting
            with_weighting: Predictions with 1.3× weighting
            ground_truth: Ground truth labels
            weighting_factor: Weighting factor used (default: 1.3)
            
        Returns:
            MetricResult with weighting effectiveness
        """
        # Convert to numpy arrays
        without_weighting = np.array(without_weighting)
        with_weighting = np.array(with_weighting)
        ground_truth = np.array(ground_truth)
        
        # Compute metrics for both
        cm_without = compute_confusion_matrix(ground_truth, without_weighting)
        cm_with = compute_confusion_matrix(ground_truth, with_weighting)
        
        metrics_without = compute_basic_metrics(cm_without)
        metrics_with = compute_basic_metrics(cm_with)
        
        # Compute improvements
        precision_improvement = metrics_with['precision'] - metrics_without['precision']
        recall_improvement = metrics_with['recall'] - metrics_without['recall']
        f1_improvement = metrics_with['f1_score'] - metrics_without['f1_score']
        
        # Overall effectiveness (average improvement)
        effectiveness = (precision_improvement + recall_improvement + f1_improvement) / 3
        
        return MetricResult(
            name=self.name,
            value=round(float(effectiveness), 4),
            metadata={
                'weighting_factor': weighting_factor,
                'num_samples': len(ground_truth)
            },
            sub_metrics={
                'effectiveness': round(float(effectiveness), 4),
                'precision_improvement': round(float(precision_improvement), 4),
                'recall_improvement': round(float(recall_improvement), 4),
                'f1_improvement': round(float(f1_improvement), 4),
                'precision_without': metrics_without['precision'],
                'precision_with': metrics_with['precision'],
                'false_positive_reduction': cm_without['false_positives'] - cm_with['false_positives']
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['without_weighting', 'with_weighting', 'ground_truth'],
            'optional': ['weighting_factor'],
            'output_type': 'float'
        }

# ============================================
# IDENTITY VERIFICATION BONUS METRIC
# ============================================

class IdentityVerificationMetric(BaseMetric):
    """
    Identity verification bonus impact metric
    
    Measures impact of 1.2× identity verification bonus.
    """
    
    def __init__(self):
        """Initialize identity verification metric"""
        super().__init__(
            name='identity_verification_bonus',
            description='Identity verification bonus impact',
            category='counter_evidence'
        )
    
    def validate_inputs(
        self,
        without_bonus: np.ndarray,
        with_bonus: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        validate_predictions(ground_truth, without_bonus)
        validate_predictions(ground_truth, with_bonus)
        return True
    
    def compute(
        self,
        without_bonus: np.ndarray,
        with_bonus: np.ndarray,
        ground_truth: np.ndarray,
        bonus_factor: float = 1.2,
        **kwargs
    ) -> MetricResult:
        """
        Compute identity verification bonus impact
        
        Args:
            without_bonus: Predictions without ID verification bonus
            with_bonus: Predictions with 1.2× bonus
            ground_truth: Ground truth labels
            bonus_factor: Bonus factor used (default: 1.2)
            
        Returns:
            MetricResult with bonus impact
        """
        # Convert to numpy arrays
        without_bonus = np.array(without_bonus)
        with_bonus = np.array(with_bonus)
        ground_truth = np.array(ground_truth)
        
        # Compute metrics
        cm_without = compute_confusion_matrix(ground_truth, without_bonus)
        cm_with = compute_confusion_matrix(ground_truth, with_bonus)
        
        metrics_without = compute_basic_metrics(cm_without)
        metrics_with = compute_basic_metrics(cm_with)
        
        # Compute impact
        precision_impact = metrics_with['precision'] - metrics_without['precision']
        recall_impact = metrics_with['recall'] - metrics_without['recall']
        
        # Overall impact
        overall_impact = (precision_impact + recall_impact) / 2
        
        return MetricResult(
            name=self.name,
            value=round(float(overall_impact), 4),
            metadata={
                'bonus_factor': bonus_factor,
                'num_samples': len(ground_truth)
            },
            sub_metrics={
                'overall_impact': round(float(overall_impact), 4),
                'precision_impact': round(float(precision_impact), 4),
                'recall_impact': round(float(recall_impact), 4),
                'accuracy_without': metrics_without['accuracy'],
                'accuracy_with': metrics_with['accuracy']
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['without_bonus', 'with_bonus', 'ground_truth'],
            'optional': ['bonus_factor'],
            'output_type': 'float'
        }

# ============================================
# COMBINED BAYESIAN METRICS
# ============================================

class BayesianMetrics:
    """
    Compute all Bayesian counter-evidence metrics
    
    Convenience class for computing all metrics at once.
    """
    
    def __init__(self):
        """Initialize all metrics"""
        self.fp_reduction_metric = FalsePositiveReductionMetric()
        self.decision_change_metric = DecisionChangeMetric()
        self.accuracy_metric = BayesianAccuracyMetric()
        self.presumption_metric = PresumptionWeightingMetric()
        self.identity_metric = IdentityVerificationMetric()
        
        logger.info("Initialized BayesianMetrics")
    
    def compute_all(
        self,
        before_predictions: np.ndarray,
        after_predictions: np.ndarray,
        ground_truth: np.ndarray,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Compute all Bayesian metrics
        
        Args:
            before_predictions: Predictions before counter-evidence
            after_predictions: Predictions after counter-evidence
            ground_truth: Ground truth labels
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Compute false positive reduction
        try:
            results['fp_reduction'] = self.fp_reduction_metric.compute(
                before_predictions, after_predictions, ground_truth
            )
        except Exception as e:
            logger.error(f"Failed to compute FP reduction: {e}")
        
        # Compute decision change rate
        try:
            results['decision_change'] = self.decision_change_metric.compute(
                before_predictions, after_predictions, ground_truth
            )
        except Exception as e:
            logger.error(f"Failed to compute decision change rate: {e}")
        
        # Compute Bayesian accuracy
        try:
            results['bayesian_accuracy'] = self.accuracy_metric.compute(
                after_predictions, ground_truth
            )
        except Exception as e:
            logger.error(f"Failed to compute Bayesian accuracy: {e}")
        
        logger.info(f"Computed {len(results)} metric(s)")
        
        return results

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def compute_fp_reduction(
    before: np.ndarray,
    after: np.ndarray,
    truth: np.ndarray
) -> float:
    """
    Compute false positive reduction
    
    Args:
        before: Predictions before counter-evidence
        after: Predictions after counter-evidence
        truth: Ground truth
        
    Returns:
        FP reduction rate
    """
    metric = FalsePositiveReductionMetric()
    result = metric.compute(before, after, truth)
    return result.value

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'FalsePositiveReductionMetric',
    'DecisionChangeMetric',
    'BayesianAccuracyMetric',
    'PresumptionWeightingMetric',
    'IdentityVerificationMetric',
    'BayesianMetrics',
    'compute_fp_reduction'
]
