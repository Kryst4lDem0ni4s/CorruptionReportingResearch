"""
Metrics Service - Performance metrics tracking with Prometheus integration

Provides:
- Request latency tracking
- Throughput monitoring
- Memory usage tracking
- Model inference time tracking
- System resource monitoring
- Metrics persistence
- Prometheus metrics export
"""

import logging
import os
import psutil
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Prometheus imports (optional - graceful degradation if not available)
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Gauge = Histogram = None

# Initialize logger
logger = logging.getLogger(__name__)


class MetricsService:
    """
    Metrics Service - Performance and system metrics tracking with Prometheus integration.

    Features:
    - Request latency tracking (percentiles)
    - Throughput monitoring (requests per time window)
    - Memory usage tracking
    - Model inference time tracking
    - Error rate monitoring
    - Metrics aggregation and persistence
    - Prometheus metrics export (if available)
    """

    def __init__(
        self,
        metrics_file: Optional[Path] = None,
        window_size: int = 1000,
        persist_interval: int = 300,
        prometheus_metrics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize metrics service.

        Args:
            metrics_file: Path to metrics persistence file
            window_size: Number of recent metrics to keep in memory
            persist_interval: Persist metrics every N seconds
            prometheus_metrics: Dict of Prometheus metric objects from main.py
        """
        self.metrics_file = metrics_file
        self.window_size = window_size
        self.persist_interval = persist_interval

        # Prometheus metrics (passed from main.py)
        self.prometheus_metrics = prometheus_metrics or {}
        self.prometheus_enabled = PROMETHEUS_AVAILABLE and bool(prometheus_metrics)

        if self.prometheus_enabled:
            logger.info("Prometheus metrics integration enabled")
        else:
            logger.info("Prometheus metrics not available - using local metrics only")

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

        # Layer-specific metrics
        self.layer_metrics = {
            'layer1_anonymity_violations': 0,
            'layer2_credibility_scores': deque(maxlen=100),
            'layer2_deepfakes_detected': 0,
            'layer3_coordination_detected': 0,
            'layer4_consensus_iterations': deque(maxlen=100),
            'layer4_convergence_times': deque(maxlen=100),
            'layer5_counter_evidence_count': 0,
            'layer5_impacts': deque(maxlen=100),
            'layer6_reports_generated': 0
        }

        # Security metrics
        self.security_metrics = {
            'hash_chain_failures': 0,
            'validation_failures': defaultdict(int),
            'crypto_failures': defaultdict(int),
            'rate_limit_blocks': defaultdict(int)
        }

        # Storage metrics
        self.storage_metrics = {
            'corruption_detected': 0,
            'index_inconsistencies': 0,
            'backup_failures': 0
        }

        # Queue metrics
        self.queue_metrics = {
            'pending_jobs': 0,
            'processing_jobs': 0,
            'job_failures': defaultdict(int)
        }

        # Research/evaluation metrics
        self.evaluation_metrics = {
            'auroc_score': 0.0,
            'precision_score': 0.0,
            'recall_score': 0.0
        }

        # System metrics
        self.process = psutil.Process(os.getpid())
        self.start_time = time.time()
        self.last_persist_time = time.time()

        # Counters
        self.total_requests = 0
        self.total_errors = 0

        logger.info("MetricsService initialized")

    # ==================== REQUEST METRICS ====================

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

            # Update Prometheus metric
            if self.prometheus_enabled and 'submission_total' in self.prometheus_metrics:
                self.prometheus_metrics['submission_total'].labels(status=status).inc()

        except Exception as e:
            logger.error(f"Failed to record submission metric: {e}")

    # ==================== LAYER-SPECIFIC METRICS ====================

    def record_anonymity_violation(self) -> None:
        """Record Layer 1 anonymity violation."""
        try:
            self.layer_metrics['layer1_anonymity_violations'] += 1
            logger.warning("Anonymity violation detected")
        except Exception as e:
            logger.error(f"Failed to record anonymity violation: {e}")

    def update_credibility_score(self, score: float) -> None:
        """
        Update Layer 2 credibility score.

        Args:
            score: Credibility score (0.0 to 1.0)
        """
        try:
            self.layer_metrics['layer2_credibility_scores'].append(score)

            # Update Prometheus gauge with average
            if self.prometheus_enabled and 'layer2_score' in self.prometheus_metrics:
                scores = self.layer_metrics['layer2_credibility_scores']
                avg_score = sum(scores) / len(scores) if scores else 0.0
                self.prometheus_metrics['layer2_score'].set(avg_score)

            logger.debug(f"Credibility score updated: {score:.3f}")

        except Exception as e:
            logger.error(f"Failed to update credibility score: {e}")

    def record_deepfake_detection(self) -> None:
        """Record Layer 2 deepfake detection."""
        try:
            self.layer_metrics['layer2_deepfakes_detected'] += 1
            logger.info("Deepfake detected")
        except Exception as e:
            logger.error(f"Failed to record deepfake detection: {e}")

    def record_coordination_detection(self) -> None:
        """Record Layer 3 coordination detection."""
        try:
            self.layer_metrics['layer3_coordination_detected'] += 1

            # Update Prometheus counter
            if self.prometheus_enabled and 'layer3_coordination' in self.prometheus_metrics:
                self.prometheus_metrics['layer3_coordination'].inc()

            logger.warning("Coordinated attack detected")

        except Exception as e:
            logger.error(f"Failed to record coordination detection: {e}")

    def update_consensus_metrics(
        self,
        iterations: int,
        convergence_time: float
    ) -> None:
        """
        Update Layer 4 consensus metrics.

        Args:
            iterations: Number of consensus iterations
            convergence_time: Time to reach consensus (seconds)
        """
        try:
            self.layer_metrics['layer4_consensus_iterations'].append(iterations)
            self.layer_metrics['layer4_convergence_times'].append(convergence_time)

            # Update Prometheus gauges
            if self.prometheus_enabled:
                if 'layer4_iterations' in self.prometheus_metrics:
                    iter_list = self.layer_metrics['layer4_consensus_iterations']
                    avg_iterations = sum(iter_list) / len(iter_list) if iter_list else 0.0
                    self.prometheus_metrics['layer4_iterations'].set(avg_iterations)

                if 'layer4_convergence' in self.prometheus_metrics:
                    self.prometheus_metrics['layer4_convergence'].set(convergence_time)

            logger.debug(
                f"Consensus metrics: {iterations} iterations, "
                f"{convergence_time:.3f}s convergence time"
            )

        except Exception as e:
            logger.error(f"Failed to update consensus metrics: {e}")

    def record_counter_evidence(self, impact_percent: float) -> None:
        """
        Record Layer 5 counter-evidence submission.

        Args:
            impact_percent: Impact percentage on credibility score
        """
        try:
            self.layer_metrics['layer5_counter_evidence_count'] += 1
            self.layer_metrics['layer5_impacts'].append(impact_percent)

            # Update Prometheus metrics
            if self.prometheus_enabled:
                if 'layer5_counter' in self.prometheus_metrics:
                    self.prometheus_metrics['layer5_counter'].inc()

                if 'layer5_impact' in self.prometheus_metrics:
                    impacts = self.layer_metrics['layer5_impacts']
                    avg_impact = sum(impacts) / len(impacts) if impacts else 0.0
                    self.prometheus_metrics['layer5_impact'].set(avg_impact)

            logger.info(f"Counter-evidence recorded: {impact_percent:.1f}% impact")

        except Exception as e:
            logger.error(f"Failed to record counter-evidence: {e}")

    def record_report_generated(self) -> None:
        """Record Layer 6 report generation."""
        try:
            self.layer_metrics['layer6_reports_generated'] += 1
            logger.info("Forensic report generated")
        except Exception as e:
            logger.error(f"Failed to record report generation: {e}")

    # ==================== SECURITY METRICS ====================

    def record_hash_chain_failure(self) -> None:
        """Record hash chain validation failure."""
        try:
            self.security_metrics['hash_chain_failures'] += 1
            logger.error("Hash chain validation failure")
        except Exception as e:
            logger.error(f"Failed to record hash chain failure: {e}")

    def record_validation_failure(self, validation_type: str) -> None:
        """
        Record input validation failure.

        Args:
            validation_type: Type of validation that failed
        """
        try:
            self.security_metrics['validation_failures'][validation_type] += 1
            logger.warning(f"Validation failure: {validation_type}")
        except Exception as e:
            logger.error(f"Failed to record validation failure: {e}")

    def record_crypto_failure(self, operation: str) -> None:
        """
        Record cryptographic operation failure.

        Args:
            operation: Crypto operation that failed
        """
        try:
            self.security_metrics['crypto_failures'][operation] += 1
            logger.error(f"Cryptographic failure: {operation}")
        except Exception as e:
            logger.error(f"Failed to record crypto failure: {e}")

    def record_rate_limit_block(self, reason: str) -> None:
        """
        Record rate limiter block.

        Args:
            reason: Reason for block
        """
        try:
            self.security_metrics['rate_limit_blocks'][reason] += 1
            logger.warning(f"Rate limit block: {reason}")
        except Exception as e:
            logger.error(f"Failed to record rate limit block: {e}")

    # ==================== STORAGE METRICS ====================

    def record_storage_corruption(self) -> None:
        """Record storage corruption detection."""
        try:
            self.storage_metrics['corruption_detected'] += 1
            logger.error("Storage corruption detected")
        except Exception as e:
            logger.error(f"Failed to record storage corruption: {e}")

    def record_index_inconsistency(self) -> None:
        """Record storage index inconsistency."""
        try:
            self.storage_metrics['index_inconsistencies'] += 1
            logger.warning("Storage index inconsistency detected")
        except Exception as e:
            logger.error(f"Failed to record index inconsistency: {e}")

    def record_backup_failure(self) -> None:
        """Record backup failure."""
        try:
            self.storage_metrics['backup_failures'] += 1
            logger.error("Backup failure")
        except Exception as e:
            logger.error(f"Failed to record backup failure: {e}")

    # ==================== QUEUE METRICS ====================

    def update_queue_metrics(
        self,
        pending: int,
        processing: int
    ) -> None:
        """
        Update queue metrics.

        Args:
            pending: Number of pending jobs
            processing: Number of processing jobs
        """
        try:
            self.queue_metrics['pending_jobs'] = pending
            self.queue_metrics['processing_jobs'] = processing

            # Update Prometheus gauges
            if self.prometheus_enabled:
                if 'queue_pending' in self.prometheus_metrics:
                    self.prometheus_metrics['queue_pending'].set(pending)
                if 'queue_processing' in self.prometheus_metrics:
                    self.prometheus_metrics['queue_processing'].set(processing)

        except Exception as e:
            logger.error(f"Failed to update queue metrics: {e}")

    def record_queue_failure(self, job_type: str) -> None:
        """
        Record queue job failure.

        Args:
            job_type: Type of job that failed
        """
        try:
            self.queue_metrics['job_failures'][job_type] += 1
            logger.error(f"Queue job failure: {job_type}")
        except Exception as e:
            logger.error(f"Failed to record queue failure: {e}")

    # ==================== EVALUATION METRICS ====================

    def update_evaluation_scores(
        self,
        auroc: Optional[float] = None,
        precision: Optional[float] = None,
        recall: Optional[float] = None
    ) -> None:
        """
        Update research evaluation scores.

        Args:
            auroc: AUROC score
            precision: Precision score
            recall: Recall score
        """
        try:
            if auroc is not None:
                self.evaluation_metrics['auroc_score'] = auroc
            if precision is not None:
                self.evaluation_metrics['precision_score'] = precision
            if recall is not None:
                self.evaluation_metrics['recall_score'] = recall

            logger.info(
                f"Evaluation scores updated: "
                f"AUROC={auroc}, Precision={precision}, Recall={recall}"
            )

        except Exception as e:
            logger.error(f"Failed to update evaluation scores: {e}")

    # ==================== METRICS RETRIEVAL ====================

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

    def get_layer_stats(self) -> Dict:
        """
        Get layer-specific statistics.

        Returns:
            dict: Layer metrics
        """
        stats = {
            'layer1': {
                'anonymity_violations': self.layer_metrics['layer1_anonymity_violations']
            },
            'layer2': {
                'deepfakes_detected': self.layer_metrics['layer2_deepfakes_detected'],
                'avg_credibility_score': round(
                    sum(self.layer_metrics['layer2_credibility_scores']) / 
                    len(self.layer_metrics['layer2_credibility_scores'])
                    if self.layer_metrics['layer2_credibility_scores'] else 0.0,
                    3
                )
            },
            'layer3': {
                'coordination_detected': self.layer_metrics['layer3_coordination_detected']
            },
            'layer4': {
                'avg_iterations': round(
                    sum(self.layer_metrics['layer4_consensus_iterations']) /
                    len(self.layer_metrics['layer4_consensus_iterations'])
                    if self.layer_metrics['layer4_consensus_iterations'] else 0.0,
                    2
                ),
                'avg_convergence_time': round(
                    sum(self.layer_metrics['layer4_convergence_times']) /
                    len(self.layer_metrics['layer4_convergence_times'])
                    if self.layer_metrics['layer4_convergence_times'] else 0.0,
                    3
                )
            },
            'layer5': {
                'counter_evidence_count': self.layer_metrics['layer5_counter_evidence_count'],
                'avg_impact_percent': round(
                    sum(self.layer_metrics['layer5_impacts']) /
                    len(self.layer_metrics['layer5_impacts'])
                    if self.layer_metrics['layer5_impacts'] else 0.0,
                    2
                )
            },
            'layer6': {
                'reports_generated': self.layer_metrics['layer6_reports_generated']
            }
        }

        return stats

    def get_security_stats(self) -> Dict:
        """Get security-related statistics."""
        return {
            'hash_chain_failures': self.security_metrics['hash_chain_failures'],
            'validation_failures': dict(self.security_metrics['validation_failures']),
            'crypto_failures': dict(self.security_metrics['crypto_failures']),
            'rate_limit_blocks': dict(self.security_metrics['rate_limit_blocks'])
        }

    def get_storage_stats(self) -> Dict:
        """Get storage-related statistics."""
        return dict(self.storage_metrics)

    def get_queue_stats(self) -> Dict:
        """Get queue-related statistics."""
        return {
            'pending_jobs': self.queue_metrics['pending_jobs'],
            'processing_jobs': self.queue_metrics['processing_jobs'],
            'job_failures': dict(self.queue_metrics['job_failures'])
        }

    def get_evaluation_stats(self) -> Dict:
        """Get research evaluation statistics."""
        return dict(self.evaluation_metrics)

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

    def get_summary(self) -> Dict:
        """
        Get summary of all metrics.

        Returns:
            dict: Summary metrics for logging/monitoring
        """
        request_stats = self.get_request_stats()
        system_metrics = self.get_system_metrics()
        layer_stats = self.get_layer_stats()

        summary = {
            'requests': request_stats.get('total_requests', 0),
            'error_rate': request_stats.get('error_rate', 0),
            'avg_latency': request_stats.get('avg_latency', 0),
            'memory_percent': system_metrics.get('memory_percent', 0),
            'cpu_percent': system_metrics.get('cpu_percent', 0),
            'uptime_hours': system_metrics.get('uptime_hours', 0),
            'submissions': self.submission_stats.get('total', 0),
            'deepfakes_detected': layer_stats['layer2']['deepfakes_detected'],
            'coordination_detected': layer_stats['layer3']['coordination_detected'],
            'reports_generated': layer_stats['layer6']['reports_generated']
        }

        return summary

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
            'layers': self.get_layer_stats(),
            'security': self.get_security_stats(),
            'storage': self.get_storage_stats(),
            'queue': self.get_queue_stats(),
            'evaluation': self.get_evaluation_stats(),
            'system': self.get_system_metrics(),
            'prometheus_enabled': self.prometheus_enabled
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

        # Reset layer metrics
        self.layer_metrics = {
            'layer1_anonymity_violations': 0,
            'layer2_credibility_scores': deque(maxlen=100),
            'layer2_deepfakes_detected': 0,
            'layer3_coordination_detected': 0,
            'layer4_consensus_iterations': deque(maxlen=100),
            'layer4_convergence_times': deque(maxlen=100),
            'layer5_counter_evidence_count': 0,
            'layer5_impacts': deque(maxlen=100),
            'layer6_reports_generated': 0
        }

        # Reset security metrics
        self.security_metrics = {
            'hash_chain_failures': 0,
            'validation_failures': defaultdict(int),
            'crypto_failures': defaultdict(int),
            'rate_limit_blocks': defaultdict(int)
        }

        # Reset storage metrics
        self.storage_metrics = {
            'corruption_detected': 0,
            'index_inconsistencies': 0,
            'backup_failures': 0
        }

        # Reset queue metrics
        self.queue_metrics = {
            'pending_jobs': 0,
            'processing_jobs': 0,
            'job_failures': defaultdict(int)
        }

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