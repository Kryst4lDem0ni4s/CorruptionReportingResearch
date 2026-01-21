"""
Performance Benchmarks - Real Implementation
Measures system performance metrics
"""

import time
import psutil
import threading
from typing import Dict, Any, List
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

import logging

logger = logging.getLogger(__name__)


class BenchmarkExperiment:
    """Performance benchmark experiment"""
    
    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        timeout: int = 300
    ):
        """Initialize experiment"""
        self.backend_url = backend_url
        self.timeout = timeout
    
    def run(self) -> Dict[str, Any]:
        """
        Run performance benchmarks
        
        Returns:
            Benchmark results
        """
        logger.info("Running performance benchmarks...")
        
        results = {}
        
        # 1. Latency test
        logger.info("Running latency test...")
        results['latency'] = self._test_latency()
        
        # 2. Throughput test
        logger.info("Running throughput test...")
        results['throughput'] = self._test_throughput()
        
        # 3. Memory usage test
        logger.info("Running memory test...")
        results['memory'] = self._test_memory()
        
        # 4. CPU usage test
        logger.info("Running CPU test...")
        results['cpu'] = self._test_cpu()
        
        # Summary
        summary = {
            'experiment': 'benchmarks',
            'results': results,
            'targets': {
                'latency_ms': 5000,
                'throughput_per_hour': 20,
                'memory_mb': 8192,
                'cpu_percent': 80
            }
        }
        
        logger.info("Benchmark Results:")
        logger.info(f"  Avg Latency: {results['latency']['avg_ms']:.2f}ms")
        logger.info(f"  Throughput: {results['throughput']['submissions_per_hour']:.1f}/hour")
        logger.info(f"  Peak Memory: {results['memory']['peak_mb']:.1f}MB")
        logger.info(f"  Avg CPU: {results['cpu']['avg_percent']:.1f}%")
        
        return summary
    
    def _test_latency(self, num_requests: int = 10) -> Dict:
        """Test API latency"""
        latencies = []
        
        for i in range(num_requests):
            try:
                start = time.time()
                response = requests.get(
                    f"{self.backend_url}/api/v1/health",
                    timeout=10
                )
                latency = (time.time() - start) * 1000  # ms
                
                if response.status_code == 200:
                    latencies.append(latency)
                    
            except Exception as e:
                logger.warning(f"Latency test request {i+1} failed: {e}")
        
        if not latencies:
            return {'error': 'All requests failed'}
        
        return {
            'num_requests': len(latencies),
            'avg_ms': round(np.mean(latencies), 2),
            'median_ms': round(np.median(latencies), 2),
            'min_ms': round(np.min(latencies), 2),
            'max_ms': round(np.max(latencies), 2),
            'p95_ms': round(np.percentile(latencies, 95), 2),
            'p99_ms': round(np.percentile(latencies, 99), 2)
        }
    
    def _test_throughput(
        self,
        num_submissions: int = 20,
        num_workers: int = 4
    ) -> Dict:
        """Test submission throughput"""
        
        start_time = time.time()
        successful = 0
        failed = 0
        
        def submit_test():
            try:
                response = requests.post(
                    f"{self.backend_url}/api/v1/submissions",
                    json={
                        'pseudonym': 'benchmark_test',
                        'description': 'Benchmark test submission',
                        'evidence_type': 'text'
                    },
                    timeout=self.timeout
                )
                return response.status_code == 200 or response.status_code == 201
            except:
                return False
        
        # Concurrent submissions
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(submit_test) for _ in range(num_submissions)]
            
            for future in as_completed(futures):
                if future.result():
                    successful += 1
                else:
                    failed += 1
        
        total_time = time.time() - start_time
        throughput_per_second = successful / total_time if total_time > 0 else 0
        throughput_per_hour = throughput_per_second * 3600
        
        return {
            'num_submissions': num_submissions,
            'successful': successful,
            'failed': failed,
            'total_time_seconds': round(total_time, 2),
            'throughput_per_second': round(throughput_per_second, 2),
            'submissions_per_hour': round(throughput_per_hour, 1),
            'avg_time_per_submission': round(total_time / successful, 2) if successful > 0 else 0
        }
    
    def _test_memory(self, duration_seconds: int = 10) -> Dict:
        """Test memory usage"""
        process = psutil.Process()
        
        memory_samples = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            memory_samples.append(memory_mb)
            time.sleep(0.5)
        
        return {
            'samples': len(memory_samples),
            'avg_mb': round(np.mean(memory_samples), 2),
            'peak_mb': round(np.max(memory_samples), 2),
            'min_mb': round(np.min(memory_samples), 2),
            'std_mb': round(np.std(memory_samples), 2)
        }
    
    def _test_cpu(self, duration_seconds: int = 10) -> Dict:
        """Test CPU usage"""
        process = psutil.Process()
        
        cpu_samples = []
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            cpu_percent = process.cpu_percent(interval=0.5)
            cpu_samples.append(cpu_percent)
        
        return {
            'samples': len(cpu_samples),
            'avg_percent': round(np.mean(cpu_samples), 2),
            'peak_percent': round(np.max(cpu_samples), 2),
            'min_percent': round(np.min(cpu_samples), 2)
        }


def run_experiment(**kwargs) -> Dict[str, Any]:
    """Convenience function to run experiment"""
    experiment = BenchmarkExperiment()
    return experiment.run(**kwargs)
