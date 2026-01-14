"""
Corruption Reporting System - Base Metric Class
Version: 1.0.0
Description: Abstract base class for evaluation metrics

This module provides:
- BaseMetric: Abstract interface for all metrics
- MetricResult: Standardized result format
- MetricRegistry: Central metric registration

Usage:
    from evaluation.metrics.base_metric import BaseMetric, MetricResult
    
    class CustomMetric(BaseMetric):
        def compute(self, y_true, y_pred, **kwargs):
            # Implement metric computation
            return MetricResult(
                name='custom_metric',
                value=0.85,
                metadata={'description': 'Custom metric'}
            )
"""

import abc
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
import logging
import numpy as np

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.metrics.base')

# ============================================
# METRIC RESULT
# ============================================

@dataclass
class MetricResult:
    """
    Standardized metric result format
    
    Attributes:
        name: Metric name
        value: Primary metric value
        metadata: Additional information
        sub_metrics: Related sub-metrics
        timestamp: Computation timestamp
        execution_time: Time taken to compute (seconds)
    """
    name: str
    value: Union[float, int, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_metrics: Dict[str, Union[float, int]] = field(default_factory=dict)
    timestamp: Optional[float] = None
    execution_time: Optional[float] = None
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def __str__(self) -> str:
        """String representation"""
        if isinstance(self.value, float):
            return f"{self.name}: {self.value:.4f}"
        else:
            return f"{self.name}: {self.value}"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"MetricResult(name='{self.name}', value={self.value})"

# ============================================
# BASE METRIC CLASS
# ============================================

class BaseMetric(abc.ABC):
    """
    Abstract base class for evaluation metrics
    
    All metric implementations should inherit from this class and implement:
    - compute(): Main metric computation
    - validate_inputs(): Input validation (optional)
    - get_requirements(): List required inputs (optional)
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        category: Optional[str] = None
    ):
        """
        Initialize metric
        
        Args:
            name: Metric name (default: class name)
            description: Metric description
            category: Metric category
        """
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self.category = category or "general"
        
        # Execution tracking
        self._last_execution_time = None
        self._call_count = 0
        
        logger.debug(f"Initialized metric: {self.name}")
    
    # ========================================
    # ABSTRACT METHODS
    # ========================================
    
    @abc.abstractmethod
    def compute(self, *args, **kwargs) -> MetricResult:
        """
        Compute metric value
        
        This method must be implemented by subclasses.
        
        Args:
            *args: Positional arguments (e.g., y_true, y_pred)
            **kwargs: Keyword arguments (e.g., threshold, weights)
            
        Returns:
            MetricResult object
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError(
            f"Metric {self.name} must implement compute() method"
        )
    
    # ========================================
    # OPTIONAL METHODS
    # ========================================
    
    def validate_inputs(self, *args, **kwargs) -> bool:
        """
        Validate metric inputs
        
        Override this method to add custom validation.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If validation fails
        """
        # Default: no validation
        return True
    
    def get_requirements(self) -> Dict[str, Any]:
        """
        Get metric requirements
        
        Returns:
            Dictionary with required inputs and their types
        """
        return {
            'inputs': [],
            'optional': [],
            'output_type': 'float'
        }
    
    def supports_confidence_interval(self) -> bool:
        """
        Check if metric supports confidence intervals
        
        Returns:
            True if supported
        """
        return False
    
    # ========================================
    # EXECUTION WRAPPER
    # ========================================
    
    def __call__(self, *args, **kwargs) -> MetricResult:
        """
        Execute metric computation with timing
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            MetricResult object
        """
        # Validate inputs
        try:
            self.validate_inputs(*args, **kwargs)
        except Exception as e:
            logger.error(f"Input validation failed for {self.name}: {e}")
            raise
        
        # Compute metric with timing
        start_time = time.time()
        
        try:
            result = self.compute(*args, **kwargs)
        except Exception as e:
            logger.error(f"Metric computation failed for {self.name}: {e}")
            raise
        
        execution_time = time.time() - start_time
        
        # Update result with execution time
        if isinstance(result, MetricResult):
            result.execution_time = execution_time
        
        # Update tracking
        self._last_execution_time = execution_time
        self._call_count += 1
        
        logger.debug(f"{self.name} computed in {execution_time:.4f}s")
        
        return result
    
    # ========================================
    # UTILITY METHODS
    # ========================================
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get metric information
        
        Returns:
            Dictionary with metric details
        """
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category,
            'requirements': self.get_requirements(),
            'supports_ci': self.supports_confidence_interval(),
            'call_count': self._call_count,
            'last_execution_time': self._last_execution_time
        }
    
    def reset_statistics(self):
        """Reset execution statistics"""
        self._call_count = 0
        self._last_execution_time = None
        logger.debug(f"Reset statistics for {self.name}")
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.name} ({self.category})"
    
    def __repr__(self) -> str:
        """Detailed representation"""
        return f"{self.__class__.__name__}(name='{self.name}', category='{self.category}')"

# ============================================
# METRIC REGISTRY
# ============================================

class MetricRegistry:
    """
    Central registry for metrics
    
    Provides:
    - Metric registration and lookup
    - Category-based organization
    - Batch metric computation
    """
    
    def __init__(self):
        """Initialize registry"""
        self._metrics: Dict[str, BaseMetric] = {}
        self._categories: Dict[str, List[str]] = {}
        
        logger.info("Initialized metric registry")
    
    # ========================================
    # REGISTRATION
    # ========================================
    
    def register(
        self,
        name: str,
        metric: BaseMetric,
        override: bool = False
    ):
        """
        Register a metric
        
        Args:
            name: Metric name
            metric: Metric instance
            override: Allow overriding existing metric
            
        Raises:
            ValueError: If metric already exists and override=False
        """
        if name in self._metrics and not override:
            raise ValueError(f"Metric '{name}' already registered")
        
        if not isinstance(metric, BaseMetric):
            raise TypeError(f"Metric must be instance of BaseMetric, got {type(metric)}")
        
        # Register metric
        self._metrics[name] = metric
        
        # Add to category
        category = metric.category
        if category not in self._categories:
            self._categories[category] = []
        if name not in self._categories[category]:
            self._categories[category].append(name)
        
        logger.info(f"Registered metric: {name} ({category})")
    
    def unregister(self, name: str):
        """
        Unregister a metric
        
        Args:
            name: Metric name
        """
        if name in self._metrics:
            metric = self._metrics[name]
            category = metric.category
            
            # Remove from registry
            del self._metrics[name]
            
            # Remove from category
            if category in self._categories:
                self._categories[category].remove(name)
                if not self._categories[category]:
                    del self._categories[category]
            
            logger.info(f"Unregistered metric: {name}")
    
    # ========================================
    # LOOKUP
    # ========================================
    
    def get(self, name: str) -> Optional[BaseMetric]:
        """
        Get metric by name
        
        Args:
            name: Metric name
            
        Returns:
            Metric instance or None
        """
        return self._metrics.get(name)
    
    def get_by_category(self, category: str) -> List[BaseMetric]:
        """
        Get all metrics in category
        
        Args:
            category: Category name
            
        Returns:
            List of metric instances
        """
        metric_names = self._categories.get(category, [])
        return [self._metrics[name] for name in metric_names]
    
    def list_metrics(self) -> List[str]:
        """List all registered metric names"""
        return list(self._metrics.keys())
    
    def list_categories(self) -> List[str]:
        """List all categories"""
        return list(self._categories.keys())
    
    # ========================================
    # BATCH COMPUTATION
    # ========================================
    
    def compute_all(
        self,
        category: Optional[str] = None,
        *args,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Compute all metrics (optionally filtered by category)
        
        Args:
            category: Filter by category (None = all)
            *args: Arguments for metric computation
            **kwargs: Keyword arguments for metric computation
            
        Returns:
            Dictionary of metric results
        """
        if category:
            metrics = self.get_by_category(category)
            metric_names = self._categories.get(category, [])
        else:
            metrics = list(self._metrics.values())
            metric_names = list(self._metrics.keys())
        
        results = {}
        
        for name, metric in zip(metric_names, metrics):
            try:
                result = metric(*args, **kwargs)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to compute {name}: {e}")
                # Continue with other metrics
        
        logger.info(f"Computed {len(results)}/{len(metrics)} metrics")
        
        return results
    
    # ========================================
    # INFORMATION
    # ========================================
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get registry information
        
        Returns:
            Dictionary with registry statistics
        """
        return {
            'total_metrics': len(self._metrics),
            'categories': {
                cat: len(names) for cat, names in self._categories.items()
            },
            'metrics': {
                name: metric.get_info() for name, metric in self._metrics.items()
            }
        }
    
    def __len__(self) -> int:
        """Number of registered metrics"""
        return len(self._metrics)
    
    def __contains__(self, name: str) -> bool:
        """Check if metric is registered"""
        return name in self._metrics
    
    def __str__(self) -> str:
        """String representation"""
        return f"MetricRegistry({len(self)} metrics, {len(self._categories)} categories)"

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'BaseMetric',
    'MetricResult',
    'MetricRegistry'
]
