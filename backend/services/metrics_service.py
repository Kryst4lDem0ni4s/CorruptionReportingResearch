"""
Metrics Service - Performance metrics tracking

Provides:
- Request latency tracking
- Throughput monitoring
- Memory usage tracking
- Model inference time tracking
- System resource monitoring
- Metrics persistence
"""

import logging
import os
import psutil
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Initialize logger
logger = logging.getLogger(__name__)


class MetricsService:
    """
    Metrics Service - Performance and system metrics tracking.
    
    Features:
    - Request latency tracking (percentiles)
    - Throughput monitoring (requests per time window)
    - Memory usage tracking
    - Model inference time tracking
    - Error rate monitoring
    - Metrics aggregation and persistence
    """
    
    def __init__(
        self,
        metrics_file: Optional[Path] = None,
        window_size: int = 1000,
        persist_interval: int = 300
    ):
        """
        Initialize metrics service.
        
        Args:
            metrics_file: Path to metrics persistence file
            window_size: Number of recent metrics to keep in memory
            persist_interval: Persist metrics every N seconds
        """
        self.metrics_file = metrics_file
        self.window_size = window_size
        self.persist_interval = persist_interval
        
        # Request metrics (sliding window)
        self.request_latencies: deque = deque(maxlen=window_size)
        self.request_timestamps: deque = deque(maxlen=window_size)
        
        # Model metrics
        self.model_inference_times: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Endpoint metrics
        self.endpoint_counts: Dict[str, int] = defaultdict(int)
        self.endpoint_errors: Dict[str, int] = defaultdict(int)
        
        # Submission metrics
        self.submission_stats = {
            'total': 0,
            'pending': 0,
            'completed': 0,
            'failed': 0
        }
        
        # System metrics
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.last_persist_time = time.time()
        
        # Counters
        self.total_requests = 0
        self.total_errors = 0
        
        logger.info("MetricsService initialized")
    
    def record_request(
        self,
        endpoint: str,
        latency: float,
        status_code: int
    ) -> None:
        """
        Record HTTP request metrics.
        
        Args:
            endpoint: API endpoint path
            latency: Request latency in seconds
            status_code: HTTP status code
        """
        try:
            # Record latency
            self.request_latencies.append(latency)
            self.request_timestamps.append(time.time())
            
            # Record endpoint
            self.endpoint_counts[endpoint] += 1
            
            # Record errors (4xx, 5xx)
            if status_code >= 400:
                self.endpoint_errors[endpoint] += 1
                self.total_errors += 1
            
            self.total_requests += 1
            
            # Check if should persist
            if time.time() - self.last_persist_time > self.persist_interval:
                self._persist_metrics()
            
        except Exception as e:
            logger.error(f"Failed to record request metric: {e}")
    
    def record_model_inference(
        self,
        model_name: str,
        inference_time: float
    ) -> None:
        """
        Record model inference time.
        
        Args:
            model_name: Name of ML model
            inference_time: Inference time in seconds
        """
        try:
            self.model_inference_times[model_name].append(inference_time)
            
            logger.debug(
                f"Model inference: {model_name} "
                f"({inference_time:.3f}s)"
            )
            
        except Exception as e:
            logger.error(f"Failed to record model metric: {e}")
    
    def record_submission(self, status: str) -> None:
        """
        Record submission status change.
        
        Args:
            status: Submission status (pending/completed/failed)
        """
        try:
            self.submission_stats['total'] += 1
            
            if status in self.submission_stats:
                self.submission_stats[status] += 1
            
        except Exception as e:
            logger.error(f"Failed to record submission metric: {e}")
    
    def get_request_stats(self) -> Dict:
        """
        Get request statistics.
        
        Returns:
            dict: Request metrics
        """
        if not self.request_latencies:
            return {
                'total_requests': 0,
                'avg_latency': 0.0,
                'p50_latency': 0.0,
                'p95_latency': 0.0,
                'p99_latency': 0.0
            }
        
        # Calculate percentiles
        sorted_latencies = sorted(self.request_latencies)
        n = len(sorted_latencies)
        
        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)
        
        stats = {
            'total_requests': self.total_requests,
            'total_errors': self.total_errors,
            'error_rate': round(self.total_errors / self.total_requests * 100, 2) if self.total_requests > 0 else 0.0,
            'recent_requests': len(self.request_latencies),
            'avg_latency': round(sum(self.request_latencies) / n, 3),
            'min_latency': round(min(self.request_latencies), 3),
            'max_latency': round(max(self.request_latencies), 3),
            'p50_latency': round(sorted_latencies[p50_idx], 3),
            'p95_latency': round(sorted_latencies[p95_idx], 3),
            'p99_latency': round(sorted_latencies[p99_idx], 3)
        }
        
        return stats
    
    def get_throughput(self, window_seconds: int = 60) -> float:
        """
        Calculate request throughput (requests per second).
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            float: Requests per second
        """
        if not self.request_timestamps:
            return 0.0
        
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        # Count requests in window
        recent_requests = sum(
            1 for ts in self.request_timestamps
            if ts > cutoff_time
        )
        
        # Calculate throughput
        throughput = recent_requests / window_seconds
        
        return round(throughput, 2)
    
    def get_model_stats(self) -> Dict:
        """
        Get model inference statistics.
        
        Returns:
            dict: Model metrics by model name
        """
        model_stats = {}
        
        for model_name, times in self.model_inference_times.items():
            if times:
                model_stats[model_name] = {
                    'total_inferences': len(times),
                    'avg_time': round(sum(times) / len(times), 3),
                    'min_time': round(min(times), 3),
                    'max_time': round(max(times), 3)
                }
        
        return model_stats
    
    def get_endpoint_stats(self) -> Dict:
        """
        Get endpoint usage statistics.
        
        Returns:
            dict: Endpoint metrics
        """
        endpoint_stats = {}
        
        for endpoint, count in self.endpoint_counts.items():
            errors = self.endpoint_errors.get(endpoint, 0)
            error_rate = (errors / count * 100) if count > 0 else 0.0
            
            endpoint_stats[endpoint] = {
                'requests': count,
                'errors': errors,
                'error_rate': round(error_rate, 2)
            }
        
        return endpoint_stats
    
    def get_system_metrics(self) -> Dict:
        """
        Get system resource metrics.
        
        Returns:
            dict: System metrics
        """
        try:
            # Memory info
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # CPU info
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # Uptime
            uptime_seconds = time.time() - self.start_time
            
            metrics = {
                'memory_rss_mb': round(memory_info.rss / 1024 / 1024, 2),
                'memory_vms_mb': round(memory_info.vms / 1024 / 1024, 2),
                'memory_percent': round(memory_percent, 2),
                'cpu_percent': round(cpu_percent, 2),
                'num_threads': self.process.num_threads(),
                'uptime_seconds': int(uptime_seconds),
                'uptime_hours': round(uptime_seconds / 3600, 2)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_all_metrics(self) -> Dict:
        """
        Get all metrics combined.
        
        Returns:
            dict: All metrics
        """
        metrics = {
            'timestamp': datetime.utcnow().isoformat(),
            'requests': self.get_request_stats(),
            'throughput_1min': self.get_throughput(60),
            'throughput_5min': self.get_throughput(300),
            'models': self.get_model_stats(),
            'endpoints': self.get_endpoint_stats(),
            'submissions': self.submission_stats.copy(),
            'system': self.get_system_metrics()
        }
        
        return metrics
    
    def _persist_metrics(self) -> None:
        """Persist metrics to file."""
        if not self.metrics_file:
            return
        
        try:
            metrics = self.get_all_metrics()
            
            # Ensure parent directory exists
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to file (JSON Lines format)
            import json
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metrics) + '\n')
            
            self.last_persist_time = time.time()
            
            logger.debug(f"Metrics persisted to {self.metrics_file}")
            
        except Exception as e:
            logger.error(f"Failed to persist metrics: {e}")
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.request_latencies.clear()
        self.request_timestamps.clear()
        self.model_inference_times.clear()
        self.endpoint_counts.clear()
        self.endpoint_errors.clear()
        self.submission_stats = {
            'total': 0,
            'pending': 0,
            'completed': 0,
            'failed': 0
        }
        self.total_requests = 0
        self.total_errors = 0
        
        logger.info("Metrics reset")
    
    def get_health_status(self) -> Dict:
        """
        Get system health status.
        
        Returns:
            dict: Health status
        """
        system_metrics = self.get_system_metrics()
        request_stats = self.get_request_stats()
        
        # Determine health status
        status = "healthy"
        issues = []
        
        # Check memory usage
        if system_metrics.get('memory_percent', 0) > 90:
            status = "unhealthy"
            issues.append("High memory usage")
        elif system_metrics.get('memory_percent', 0) > 75:
            status = "degraded"
            issues.append("Elevated memory usage")
        
        # Check error rate
        error_rate = request_stats.get('error_rate', 0)
        if error_rate > 10:
            status = "unhealthy"
            issues.append("High error rate")
        elif error_rate > 5:
            if status == "healthy":
                status = "degraded"
            issues.append("Elevated error rate")
        
        health = {
            'status': status,
            'timestamp': datetime.utcnow().isoformat(),
            'uptime_hours': system_metrics.get('uptime_hours', 0),
            'issues': issues,
            'metrics': {
                'memory_percent': system_metrics.get('memory_percent', 0),
                'cpu_percent': system_metrics.get('cpu_percent', 0),
                'error_rate': error_rate,
                'avg_latency': request_stats.get('avg_latency', 0)
            }
        }
        
        return health
