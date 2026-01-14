"""
Corruption Reporting System - Coordination Detection Metrics
Version: 1.0.0
Description: Metrics for evaluating coordination attack detection

This module provides:
- Precision, Recall, F1 for coordination detection
- Graph-based detection accuracy
- Community detection performance
- Attack pattern recognition metrics

Usage:
    from evaluation.metrics.coordination_metrics import (
        CoordinationDetectionMetric, GraphMetrics
    )
    
    # Compute detection metrics
    metric = CoordinationDetectionMetric()
    result = metric.compute(y_true, y_pred)
    
    # Analyze graph properties
    graph_metrics = GraphMetrics()
    graph_result = graph_metrics.compute(graph, ground_truth_communities)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
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

# Try to import networkx
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    logger = logging.getLogger('evaluation.metrics.coordination')
    logger.warning("NetworkX not available, some features disabled")

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.metrics.coordination')

# ============================================
# COORDINATION DETECTION METRIC
# ============================================

class CoordinationDetectionMetric(BaseMetric):
    """
    Coordination attack detection metric
    
    Measures ability to detect coordinated attacks in submissions.
    
    Target: Precision ≥0.70, Recall ≥0.70
    """
    
    def __init__(self):
        """Initialize coordination detection metric"""
        super().__init__(
            name='coordination_detection',
            description='Coordination attack detection performance',
            category='coordination_detection'
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
        Compute coordination detection metrics
        
        Args:
            y_true: Ground truth labels (0=normal, 1=coordinated)
            y_pred: Predicted labels (0=normal, 1=coordinated)
            
        Returns:
            MetricResult with detection metrics
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred)
        
        # Compute basic metrics
        metrics = compute_basic_metrics(cm)
        
        # Check targets
        meets_precision_target = metrics['precision'] >= 0.70
        meets_recall_target = metrics['recall'] >= 0.70
        meets_targets = meets_precision_target and meets_recall_target
        
        return MetricResult(
            name=self.name,
            value=metrics['f1_score'],
            metadata={
                'confusion_matrix': cm,
                'target_precision': 0.70,
                'target_recall': 0.70,
                'meets_precision_target': meets_precision_target,
                'meets_recall_target': meets_recall_target,
                'meets_all_targets': meets_targets,
                'num_samples': cm['total'],
                'num_coordinated': int(np.sum(y_true == 1)),
                'num_normal': int(np.sum(y_true == 0))
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
            'output_type': 'float'
        }

# ============================================
# PATTERN DETECTION METRIC
# ============================================

class PatternDetectionMetric(BaseMetric):
    """
    Attack pattern detection metric
    
    Measures ability to detect specific coordination patterns:
    - Linguistic similarity
    - Temporal clustering
    - Style consistency
    - etc.
    """
    
    def __init__(self):
        """Initialize pattern detection metric"""
        super().__init__(
            name='pattern_detection',
            description='Attack pattern detection performance',
            category='coordination_detection'
        )
    
    def validate_inputs(
        self,
        pattern_predictions: Dict[str, List[bool]],
        pattern_ground_truth: Dict[str, List[bool]],
        **kwargs
    ) -> bool:
        """
        Validate inputs
        
        Args:
            pattern_predictions: Dict mapping pattern -> predictions
            pattern_ground_truth: Dict mapping pattern -> ground truth
            
        Returns:
            True if valid
        """
        if not pattern_predictions or not pattern_ground_truth:
            raise ValueError("Pattern predictions and ground truth required")
        
        # Check that patterns match
        pred_patterns = set(pattern_predictions.keys())
        truth_patterns = set(pattern_ground_truth.keys())
        
        if pred_patterns != truth_patterns:
            raise ValueError(
                f"Pattern mismatch: {pred_patterns} vs {truth_patterns}"
            )
        
        return True
    
    def compute(
        self,
        pattern_predictions: Dict[str, List[bool]],
        pattern_ground_truth: Dict[str, List[bool]],
        **kwargs
    ) -> MetricResult:
        """
        Compute pattern detection metrics
        
        Args:
            pattern_predictions: Dict mapping pattern name -> predictions
            pattern_ground_truth: Dict mapping pattern name -> ground truth
            
        Returns:
            MetricResult with per-pattern metrics
        """
        pattern_metrics = {}
        overall_scores = []
        
        for pattern_name in pattern_predictions.keys():
            y_pred = np.array(pattern_predictions[pattern_name])
            y_true = np.array(pattern_ground_truth[pattern_name])
            
            # Compute metrics for this pattern
            cm = compute_confusion_matrix(y_true, y_pred)
            metrics = compute_basic_metrics(cm)
            
            pattern_metrics[pattern_name] = {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'num_samples': cm['total']
            }
            
            overall_scores.append(metrics['f1_score'])
        
        # Compute average F1 across patterns
        avg_f1 = np.mean(overall_scores) if overall_scores else 0.0
        
        return MetricResult(
            name=self.name,
            value=round(float(avg_f1), 4),
            metadata={
                'num_patterns': len(pattern_metrics),
                'patterns': list(pattern_metrics.keys())
            },
            sub_metrics=pattern_metrics
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['pattern_predictions', 'pattern_ground_truth'],
            'optional': [],
            'output_type': 'float'
        }

# ============================================
# GRAPH METRICS
# ============================================

class GraphMetrics(BaseMetric):
    """
    Graph-based coordination detection metrics
    
    Analyzes graph properties and community detection performance.
    """
    
    def __init__(self):
        """Initialize graph metrics"""
        super().__init__(
            name='graph_metrics',
            description='Graph-based detection metrics',
            category='coordination_detection'
        )
        
        if not HAS_NETWORKX:
            logger.warning("NetworkX not available, limited functionality")
    
    def validate_inputs(self, graph: Any, **kwargs) -> bool:
        """Validate graph input"""
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for graph metrics")
        
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Expected nx.Graph, got {type(graph)}")
        
        return True
    
    def compute(
        self,
        graph: 'nx.Graph',
        ground_truth_communities: Optional[List[Set[int]]] = None,
        detected_communities: Optional[List[Set[int]]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute graph metrics
        
        Args:
            graph: NetworkX graph
            ground_truth_communities: Ground truth communities (optional)
            detected_communities: Detected communities (optional)
            
        Returns:
            MetricResult with graph metrics
        """
        if not HAS_NETWORKX:
            raise ImportError("NetworkX required for graph metrics")
        
        # Basic graph properties
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        density = nx.density(graph)
        
        # Connected components
        num_components = nx.number_connected_components(graph)
        largest_component_size = len(max(nx.connected_components(graph), key=len)) if num_nodes > 0 else 0
        
        # Clustering coefficient
        avg_clustering = nx.average_clustering(graph) if num_nodes > 0 else 0
        
        sub_metrics = {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'density': round(density, 4),
            'num_components': num_components,
            'largest_component_size': largest_component_size,
            'avg_clustering': round(avg_clustering, 4)
        }
        
        # Community detection evaluation
        if ground_truth_communities and detected_communities:
            nmi_score = self._compute_nmi(
                ground_truth_communities,
                detected_communities,
                num_nodes
            )
            sub_metrics['nmi'] = round(nmi_score, 4)
        
        return MetricResult(
            name=self.name,
            value=density,  # Use density as primary value
            metadata={
                'graph_type': type(graph).__name__,
                'has_community_eval': bool(ground_truth_communities and detected_communities)
            },
            sub_metrics=sub_metrics
        )
    
    def _compute_nmi(
        self,
        communities_true: List[Set[int]],
        communities_pred: List[Set[int]],
        num_nodes: int
    ) -> float:
        """
        Compute Normalized Mutual Information
        
        Args:
            communities_true: Ground truth communities
            communities_pred: Predicted communities
            num_nodes: Total number of nodes
            
        Returns:
            NMI score
        """
        # Convert communities to labels
        labels_true = self._communities_to_labels(communities_true, num_nodes)
        labels_pred = self._communities_to_labels(communities_pred, num_nodes)
        
        # Compute NMI (simplified implementation)
        # For production, use sklearn.metrics.normalized_mutual_info_score
        # This is a basic implementation
        
        from collections import Counter
        
        # Mutual information
        n = len(labels_true)
        
        # Count co-occurrences
        contingency = {}
        for i in range(n):
            key = (labels_true[i], labels_pred[i])
            contingency[key] = contingency.get(key, 0) + 1
        
        # Compute MI (simplified)
        mi = 0.0
        for (t, p), count in contingency.items():
            if count > 0:
                mi += (count / n) * np.log2((count * n) / (sum(1 for lt in labels_true if lt == t) * sum(1 for lp in labels_pred if lp == p)))
        
        # Normalize
        h_true = -sum((c / n) * np.log2(c / n) for c in Counter(labels_true).values())
        h_pred = -sum((c / n) * np.log2(c / n) for c in Counter(labels_pred).values())
        
        nmi = mi / np.sqrt(h_true * h_pred) if h_true > 0 and h_pred > 0 else 0.0
        
        return nmi
    
    def _communities_to_labels(
        self,
        communities: List[Set[int]],
        num_nodes: int
    ) -> List[int]:
        """Convert community sets to label array"""
        labels = [-1] * num_nodes
        
        for comm_id, community in enumerate(communities):
            for node in community:
                if node < num_nodes:
                    labels[node] = comm_id
        
        return labels
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['graph'],
            'optional': ['ground_truth_communities', 'detected_communities'],
            'output_type': 'float'
        }

# ============================================
# FALSE POSITIVE RATE METRIC
# ============================================

class FalsePositiveRateMetric(BaseMetric):
    """
    False positive rate for coordination detection
    
    Measures rate of incorrectly flagging normal submissions as coordinated.
    """
    
    def __init__(self):
        """Initialize false positive rate metric"""
        super().__init__(
            name='false_positive_rate',
            description='Rate of false coordination alerts',
            category='coordination_detection'
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
        Compute false positive rate
        
        Args:
            y_true: Ground truth labels (0=normal, 1=coordinated)
            y_pred: Predicted labels (0=normal, 1=coordinated)
            
        Returns:
            MetricResult with FPR
        """
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Compute confusion matrix
        cm = compute_confusion_matrix(y_true, y_pred)
        
        # Compute FPR
        fp = cm['false_positives']
        tn = cm['true_negatives']
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Compute related metrics
        fnr = cm['false_negatives'] / (cm['false_negatives'] + cm['true_positives']) if (cm['false_negatives'] + cm['true_positives']) > 0 else 0.0
        
        return MetricResult(
            name=self.name,
            value=round(float(fpr), 4),
            metadata={
                'num_false_positives': fp,
                'num_true_negatives': tn,
                'target': 'minimize',
                'acceptable_threshold': 0.10  # 10% FPR acceptable
            },
            sub_metrics={
                'false_positive_rate': round(float(fpr), 4),
                'false_negative_rate': round(float(fnr), 4),
                'true_negative_rate': round(float(tn / (tn + fp) if (tn + fp) > 0 else 0), 4)
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
# COMBINED COORDINATION METRICS
# ============================================

class CoordinationMetrics:
    """
    Compute all coordination detection metrics
    
    Convenience class for computing all metrics at once.
    """
    
    def __init__(self):
        """Initialize all metrics"""
        self.detection_metric = CoordinationDetectionMetric()
        self.fpr_metric = FalsePositiveRateMetric()
        
        if HAS_NETWORKX:
            self.graph_metric = GraphMetrics()
        else:
            self.graph_metric = None
            logger.warning("Graph metrics not available (NetworkX required)")
        
        logger.info("Initialized CoordinationMetrics")
    
    def compute_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        graph: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Compute all coordination metrics
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            graph: NetworkX graph (optional)
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Compute detection metrics
        try:
            results['detection'] = self.detection_metric.compute(y_true, y_pred)
        except Exception as e:
            logger.error(f"Failed to compute detection metrics: {e}")
        
        # Compute FPR
        try:
            results['false_positive_rate'] = self.fpr_metric.compute(y_true, y_pred)
        except Exception as e:
            logger.error(f"Failed to compute FPR: {e}")
        
        # Compute graph metrics if graph provided
        if graph and self.graph_metric:
            try:
                results['graph'] = self.graph_metric.compute(graph, **kwargs)
            except Exception as e:
                logger.error(f"Failed to compute graph metrics: {e}")
        
        logger.info(f"Computed {len(results)} metric(s)")
        
        return results

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def compute_coordination_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute coordination detection metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    metric = CoordinationDetectionMetric()
    result = metric.compute(y_true, y_pred)
    return result.sub_metrics

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'CoordinationDetectionMetric',
    'PatternDetectionMetric',
    'GraphMetrics',
    'FalsePositiveRateMetric',
    'CoordinationMetrics',
    'compute_coordination_metrics',
    'HAS_NETWORKX'
]
