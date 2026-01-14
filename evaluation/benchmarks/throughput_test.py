"""
Corruption Reporting System - Throughput Benchmarks
Version: 1.0.0
Description: Measure concurrent load handling and throughput

This module provides:
- Concurrent submission testing
- Requests per second measurement
- Load pattern simulation
- Bottleneck identification
- Scalability analysis

Usage:
    from evaluation.benchmarks.throughput_test import run_throughput_test
    
    results = run_throughput_test(num_concurrent=10, duration=60)
    print(f"Throughput: {results['requests_per_second']:.2f} req/s")
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.benchmarks.throughput')

# ============================================
# THREAD-SAFE RESULT COLLECTOR
# ============================================

class ResultCollector:
    """Thread-safe result collection"""
    
    def __init__(self):
        self.lock = Lock()
        self.results = []
        self.errors = []
        self.start_time = None
        self.end_time = None
    
    def add_result(self, result: Dict[str, Any]):
        """Add successful result"""
        with self.lock:
            self.results.append(result)
    
    def add_error(self, error: Dict[str, Any]):
        """Add error"""
        with self.lock:
            self.errors.append(error)
    
    def set_start_time(self):
        """Set benchmark start time"""
        self.start_time = time.time()
    
    def set_end_time(self):
        """Set benchmark end time"""
        self.end_time = time.time()
    
    def get_duration(self) -> float:
        """Get total duration in seconds"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        with self.lock:
            total_requests = len(self.results) + len(self.errors)
            successful_requests = len(self.results)
            failed_requests = len(self.errors)
            duration = self.get_duration()
            
            # Calculate latencies
            latencies = [r.get('latency', 0) for r in self.results]
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0,
                'duration': duration,
                'requests_per_second': successful_requests / duration if duration > 0 else 0,
                'latencies': {
                    'mean': float(np.mean(latencies)) if latencies else 0,
                    'median': float(np.median(latencies)) if latencies else 0,
                    'min': float(np.min(latencies)) if latencies else 0,
                    'max': float(np.max(latencies)) if latencies else 0,
                    'p95': float(np.percentile(latencies, 95)) if latencies else 0,
                    'p99': float(np.percentile(latencies, 99)) if latencies else 0
                }
            }

# ============================================
# WORKLOAD GENERATORS
# ============================================

def generate_test_submission(index: int) -> Dict[str, Any]:
    """
    Generate test submission data
    
    Args:
        index: Submission index
        
    Returns:
        Test submission dictionary
    """
    return {
        'submission_id': f'throughput_test_{index}',
        'evidence_type': 'text',
        'narrative': f'Throughput test submission #{index}',
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'metadata': {
            'test': True,
            'benchmark': 'throughput',
            'index': index
        }
    }

def submit_task(
    submission_func: Callable,
    submission_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute single submission task
    
    Args:
        submission_func: Function to submit
        submission_data: Submission data
        
    Returns:
        Result dictionary with timing
    """
    start_time = time.time()
    
    try:
        result = submission_func(submission_data)
        latency = time.time() - start_time
        
        return {
            'success': True,
            'latency': latency,
            'submission_id': submission_data.get('submission_id'),
            'result': result
        }
    
    except Exception as e:
        latency = time.time() - start_time
        
        return {
            'success': False,
            'latency': latency,
            'submission_id': submission_data.get('submission_id'),
            'error': str(e)
        }

# ============================================
# THROUGHPUT BENCHMARK CLASS
# ============================================

@dataclass
class ThroughputBenchmark:
    """Throughput benchmark configuration and results"""
    
    num_concurrent: int = 10
    total_requests: int = 100
    ramp_up_time: float = 0.0
    test_duration: Optional[float] = None
    
    # Results
    collector: ResultCollector = field(default_factory=ResultCollector)
    
    def get_results(self) -> Dict[str, Any]:
        """Get benchmark results"""
        return self.collector.get_summary()

# ============================================
# CONCURRENT LOAD TESTING
# ============================================

def run_concurrent_load(
    num_concurrent: int = 10,
    total_requests: int = 100,
    ramp_up_time: float = 0.0
) -> Dict[str, Any]:
    """
    Run concurrent load test
    
    Args:
        num_concurrent: Number of concurrent workers
        total_requests: Total number of requests
        ramp_up_time: Ramp-up period in seconds
        
    Returns:
        Throughput metrics
    """
    logger.info(f"Starting concurrent load test: {num_concurrent} workers, {total_requests} requests")
    
    collector = ResultCollector()
    collector.set_start_time()
    
    # Import submission function
    try:
        from backend.core.orchestrator import SubmissionOrchestrator
        orchestrator = SubmissionOrchestrator()
        submission_func = orchestrator.process_submission
    except Exception as e:
        logger.error(f"Failed to import orchestrator: {e}")
        return {'error': str(e)}
    
    # Generate submissions
    submissions = [generate_test_submission(i) for i in range(total_requests)]
    
    # Calculate ramp-up delay per worker
    worker_delay = ramp_up_time / num_concurrent if num_concurrent > 0 else 0
    
    # Execute concurrent requests
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = []
        
        for i, submission in enumerate(submissions):
            # Apply ramp-up delay
            if worker_delay > 0 and i < num_concurrent:
                time.sleep(worker_delay)
            
            future = executor.submit(submit_task, submission_func, submission)
            futures.append(future)
        
        # Collect results
        for future in as_completed(futures):
            try:
                result = future.result()
                
                if result['success']:
                    collector.add_result(result)
                else:
                    collector.add_error(result)
            
            except Exception as e:
                collector.add_error({'error': str(e)})
    
    collector.set_end_time()
    
    # Get summary
    summary = collector.get_summary()
    summary['configuration'] = {
        'num_concurrent': num_concurrent,
        'total_requests': total_requests,
        'ramp_up_time': ramp_up_time
    }
    
    logger.info(f"Concurrent load test completed: {summary['requests_per_second']:.2f} req/s")
    
    return summary

# ============================================
# SUSTAINED LOAD TESTING
# ============================================

def run_sustained_load(
    num_concurrent: int = 5,
    duration: float = 60.0,
    target_rps: Optional[float] = None
) -> Dict[str, Any]:
    """
    Run sustained load test for specified duration
    
    Args:
        num_concurrent: Number of concurrent workers
        duration: Test duration in seconds
        target_rps: Target requests per second (None = unlimited)
        
    Returns:
        Throughput metrics
    """
    logger.info(f"Starting sustained load test: {duration}s duration, {num_concurrent} workers")
    
    collector = ResultCollector()
    collector.set_start_time()
    
    # Import submission function
    try:
        from backend.core.orchestrator import SubmissionOrchestrator
        orchestrator = SubmissionOrchestrator()
        submission_func = orchestrator.process_submission
    except Exception as e:
        logger.error(f"Failed to import orchestrator: {e}")
        return {'error': str(e)}
    
    # Calculate inter-request delay if target RPS specified
    request_delay = 1.0 / target_rps if target_rps else 0
    
    # Run until duration expires
    end_time = time.time() + duration
    request_count = 0
    
    with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
        futures = []
        
        while time.time() < end_time:
            # Generate submission
            submission = generate_test_submission(request_count)
            
            # Submit task
            future = executor.submit(submit_task, submission_func, submission)
            futures.append(future)
            
            request_count += 1
            
            # Apply rate limiting
            if request_delay > 0:
                time.sleep(request_delay)
        
        # Wait for completion
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                
                if result['success']:
                    collector.add_result(result)
                else:
                    collector.add_error(result)
            
            except Exception as e:
                collector.add_error({'error': str(e)})
    
    collector.set_end_time()
    
    # Get summary
    summary = collector.get_summary()
    summary['configuration'] = {
        'num_concurrent': num_concurrent,
        'duration': duration,
        'target_rps': target_rps
    }
    
    logger.info(f"Sustained load test completed: {summary['requests_per_second']:.2f} req/s")
    
    return summary

# ============================================
# SPIKE TESTING
# ============================================

def run_spike_test(
    baseline_concurrent: int = 5,
    spike_concurrent: int = 20,
    baseline_duration: float = 30.0,
    spike_duration: float = 10.0
) -> Dict[str, Any]:
    """
    Run spike test (sudden increase in load)
    
    Args:
        baseline_concurrent: Baseline concurrent users
        spike_concurrent: Spike concurrent users
        baseline_duration: Baseline duration
        spike_duration: Spike duration
        
    Returns:
        Throughput metrics for baseline and spike
    """
    logger.info("Starting spike test...")
    
    results = {}
    
    # Baseline load
    logger.info(f"Baseline phase: {baseline_concurrent} users, {baseline_duration}s")
    baseline_results = run_sustained_load(
        num_concurrent=baseline_concurrent,
        duration=baseline_duration
    )
    results['baseline'] = baseline_results
    
    # Spike load
    logger.info(f"Spike phase: {spike_concurrent} users, {spike_duration}s")
    spike_results = run_sustained_load(
        num_concurrent=spike_concurrent,
        duration=spike_duration
    )
    results['spike'] = spike_results
    
    # Recovery phase (back to baseline)
    logger.info(f"Recovery phase: {baseline_concurrent} users, {baseline_duration}s")
    recovery_results = run_sustained_load(
        num_concurrent=baseline_concurrent,
        duration=baseline_duration
    )
    results['recovery'] = recovery_results
    
    logger.info("Spike test completed")
    
    return results

# ============================================
# MAIN BENCHMARK FUNCTION
# ============================================

def run_throughput_test(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive throughput benchmarks
    
    Args:
        config: Benchmark configuration
            - test_type: 'concurrent', 'sustained', 'spike' (default: 'concurrent')
            - num_concurrent: Concurrent workers (default: 10)
            - total_requests: Total requests for concurrent test (default: 100)
            - duration: Duration for sustained test (default: 60)
            - ramp_up_time: Ramp-up period (default: 0)
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Starting throughput benchmarks...")
    
    # Parse configuration
    config = config or {}
    test_type = config.get('test_type', 'concurrent')
    num_concurrent = config.get('num_concurrent', 10)
    total_requests = config.get('total_requests', 100)
    duration = config.get('duration', 60.0)
    ramp_up_time = config.get('ramp_up_time', 0.0)
    
    results = {
        'configuration': config,
        'test_type': test_type
    }
    
    # Run appropriate test
    if test_type == 'concurrent':
        test_results = run_concurrent_load(
            num_concurrent=num_concurrent,
            total_requests=total_requests,
            ramp_up_time=ramp_up_time
        )
        results.update(test_results)
    
    elif test_type == 'sustained':
        test_results = run_sustained_load(
            num_concurrent=num_concurrent,
            duration=duration
        )
        results.update(test_results)
    
    elif test_type == 'spike':
        baseline = config.get('baseline_concurrent', 5)
        spike = config.get('spike_concurrent', 20)
        baseline_dur = config.get('baseline_duration', 30.0)
        spike_dur = config.get('spike_duration', 10.0)
        
        test_results = run_spike_test(
            baseline_concurrent=baseline,
            spike_concurrent=spike,
            baseline_duration=baseline_dur,
            spike_duration=spike_dur
        )
        results.update(test_results)
    
    else:
        raise ValueError(f"Unknown test type: {test_type}")
    
    logger.info("Throughput benchmarks completed")
    
    return results

# ============================================
# COMMAND LINE INTERFACE
# ============================================

def main():
    """Command line interface for throughput benchmarks"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Run throughput benchmarks')
    parser.add_argument('--type', type=str, default='concurrent',
                        choices=['concurrent', 'sustained', 'spike'],
                        help='Test type')
    parser.add_argument('--concurrent', type=int, default=10,
                        help='Number of concurrent workers')
    parser.add_argument('--requests', type=int, default=100,
                        help='Total requests (concurrent test)')
    parser.add_argument('--duration', type=float, default=60.0,
                        help='Duration in seconds (sustained test)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    
    # Run benchmarks
    config = {
        'test_type': args.type,
        'num_concurrent': args.concurrent,
        'total_requests': args.requests,
        'duration': args.duration
    }
    
    results = run_throughput_test(config)
    
    # Print results
    print("\n" + "="*70)
    print("THROUGHPUT BENCHMARK RESULTS")
    print("="*70)
    
    if args.type == 'spike':
        for phase in ['baseline', 'spike', 'recovery']:
            if phase in results:
                phase_data = results[phase]
                print(f"\n{phase.upper()} PHASE:")
                print(f"  Requests/Second:     {phase_data.get('requests_per_second', 0):.2f}")
                print(f"  Success Rate:        {phase_data.get('success_rate', 0)*100:.1f}%")
                print(f"  Mean Latency:        {phase_data.get('latencies', {}).get('mean', 0):.3f}s")
    else:
        print(f"\nRequests/Second:       {results.get('requests_per_second', 0):.2f}")
        print(f"Total Requests:        {results.get('total_requests', 0)}")
        print(f"Successful:            {results.get('successful_requests', 0)}")
        print(f"Failed:                {results.get('failed_requests', 0)}")
        print(f"Success Rate:          {results.get('success_rate', 0)*100:.1f}%")
        print(f"Duration:              {results.get('duration', 0):.2f}s")
        
        if 'latencies' in results:
            lat = results['latencies']
            print(f"\nLatency Statistics:")
            print(f"  Mean:                {lat.get('mean', 0):.3f}s")
            print(f"  Median:              {lat.get('median', 0):.3f}s")
            print(f"  Min:                 {lat.get('min', 0):.3f}s")
            print(f"  Max:                 {lat.get('max', 0):.3f}s")
            print(f"  P95:                 {lat.get('p95', 0):.3f}s")
            print(f"  P99:                 {lat.get('p99', 0):.3f}s")
    
    print("="*70 + "\n")
    
    # Save to file
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}\n")

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'run_throughput_test',
    'run_concurrent_load',
    'run_sustained_load',
    'run_spike_test',
    'ThroughputBenchmark',
    'ResultCollector'
]

if __name__ == '__main__':
    main()
