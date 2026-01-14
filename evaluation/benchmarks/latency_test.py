"""
Corruption Reporting System - Latency Benchmarks
Version: 1.0.0
Description: Measure processing time for system components

This module provides:
- End-to-end submission latency
- Layer-by-layer timing
- Model inference timing
- Storage operation timing
- Statistical analysis

Usage:
    from evaluation.benchmarks.latency_test import run_latency_test
    
    results = run_latency_test(num_iterations=10)
    print(results['end_to_end']['mean'])
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.benchmarks.latency')

# ============================================
# TIMING UTILITIES
# ============================================

class Timer:
    """Context manager for timing code blocks"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time
    
    def get_elapsed(self) -> float:
        """Get elapsed time in seconds"""
        return self.elapsed if self.elapsed is not None else 0.0

def measure_execution_time(func: Callable, *args, **kwargs) -> tuple:
    """
    Measure function execution time
    
    Args:
        func: Function to measure
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, elapsed_time)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed

# ============================================
# STATISTICS
# ============================================

def compute_timing_stats(times: List[float]) -> Dict[str, float]:
    """
    Compute statistics for timing measurements
    
    Args:
        times: List of timing measurements
        
    Returns:
        Dictionary with statistics
    """
    times_array = np.array(times)
    
    return {
        'mean': float(np.mean(times_array)),
        'median': float(np.median(times_array)),
        'std': float(np.std(times_array)),
        'min': float(np.min(times_array)),
        'max': float(np.max(times_array)),
        'p50': float(np.percentile(times_array, 50)),
        'p95': float(np.percentile(times_array, 95)),
        'p99': float(np.percentile(times_array, 99)),
        'count': len(times)
    }

# ============================================
# LATENCY BENCHMARK CLASS
# ============================================

@dataclass
class LatencyBenchmark:
    """Latency benchmark configuration and results"""
    
    num_iterations: int = 10
    warmup_iterations: int = 2
    test_data_path: Optional[str] = None
    
    # Results
    end_to_end_times: List[float] = field(default_factory=list)
    layer_times: Dict[str, List[float]] = field(default_factory=dict)
    model_times: Dict[str, List[float]] = field(default_factory=dict)
    storage_times: Dict[str, List[float]] = field(default_factory=dict)
    
    def add_end_to_end_time(self, elapsed: float):
        """Add end-to-end timing measurement"""
        self.end_to_end_times.append(elapsed)
    
    def add_layer_time(self, layer_name: str, elapsed: float):
        """Add layer timing measurement"""
        if layer_name not in self.layer_times:
            self.layer_times[layer_name] = []
        self.layer_times[layer_name].append(elapsed)
    
    def add_model_time(self, model_name: str, elapsed: float):
        """Add model timing measurement"""
        if model_name not in self.model_times:
            self.model_times[model_name] = []
        self.model_times[model_name].append(elapsed)
    
    def add_storage_time(self, operation: str, elapsed: float):
        """Add storage timing measurement"""
        if operation not in self.storage_times:
            self.storage_times[operation] = []
        self.storage_times[operation].append(elapsed)
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get benchmark results with statistics
        
        Returns:
            Dictionary with results
        """
        results = {}
        
        # End-to-end statistics
        if self.end_to_end_times:
            results['end_to_end'] = compute_timing_stats(self.end_to_end_times)
        
        # Layer statistics
        if self.layer_times:
            results['layers'] = {
                layer: compute_timing_stats(times)
                for layer, times in self.layer_times.items()
            }
        
        # Model statistics
        if self.model_times:
            results['models'] = {
                model: compute_timing_stats(times)
                for model, times in self.model_times.items()
            }
        
        # Storage statistics
        if self.storage_times:
            results['storage'] = {
                op: compute_timing_stats(times)
                for op, times in self.storage_times.items()
            }
        
        return results

# ============================================
# LAYER LATENCY TESTING
# ============================================

def measure_layer_latency(
    layer_name: str,
    test_input: Any,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Measure latency for a specific layer
    
    Args:
        layer_name: Name of the layer
        test_input: Test input data
        num_iterations: Number of iterations
        
    Returns:
        Timing statistics
    """
    try:
        # Import layer dynamically
        if layer_name == 'layer1_anonymity':
            from backend.core.layer1_anonymity import AnonymityLayer
            layer = AnonymityLayer()
            func = layer.process_submission
        elif layer_name == 'layer2_credibility':
            from backend.core.layer2_credibility import CredibilityLayer
            layer = CredibilityLayer()
            func = layer.assess_credibility
        elif layer_name == 'layer3_coordination':
            from backend.core.layer3_coordination import CoordinationLayer
            layer = CoordinationLayer()
            func = layer.detect_coordination
        elif layer_name == 'layer4_consensus':
            from backend.core.layer4_consensus import ConsensusLayer
            layer = ConsensusLayer()
            func = layer.simulate_consensus
        elif layer_name == 'layer5_counter_evidence':
            from backend.core.layer5_counter_evidence import CounterEvidenceLayer
            layer = CounterEvidenceLayer()
            func = layer.process_counter_evidence
        elif layer_name == 'layer6_reporting':
            from backend.core.layer6_reporting import ReportingLayer
            layer = ReportingLayer()
            func = layer.generate_report
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        # Measure timing
        times = []
        for _ in range(num_iterations):
            with Timer() as t:
                try:
                    func(test_input)
                except Exception as e:
                    logger.warning(f"Layer execution failed: {e}")
            times.append(t.get_elapsed())
        
        return compute_timing_stats(times)
    
    except Exception as e:
        logger.error(f"Failed to measure layer {layer_name}: {e}")
        return {'error': str(e)}

# ============================================
# MODEL LATENCY TESTING
# ============================================

def measure_model_latency(
    model_name: str,
    test_input: Any,
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Measure model inference latency
    
    Args:
        model_name: Name of the model
        test_input: Test input data
        num_iterations: Number of iterations
        
    Returns:
        Timing statistics
    """
    try:
        # Import model dynamically
        if model_name == 'clip':
            from backend.models.clip_model import CLIPModel
            model = CLIPModel()
            func = model.predict
        elif model_name == 'wav2vec':
            from backend.models.wav2vec_model import Wav2VecModel
            model = Wav2VecModel()
            func = model.extract_features
        elif model_name == 'blip':
            from backend.models.blip_model import BLIPModel
            model = BLIPModel()
            func = model.generate_caption
        elif model_name == 'sentence_transformer':
            from backend.models.sentence_transformer import SentenceTransformerModel
            model = SentenceTransformerModel()
            func = model.encode
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Warmup
        for _ in range(2):
            try:
                func(test_input)
            except:
                pass
        
        # Measure timing
        times = []
        for _ in range(num_iterations):
            with Timer() as t:
                try:
                    func(test_input)
                except Exception as e:
                    logger.warning(f"Model inference failed: {e}")
            times.append(t.get_elapsed())
        
        return compute_timing_stats(times)
    
    except Exception as e:
        logger.error(f"Failed to measure model {model_name}: {e}")
        return {'error': str(e)}

# ============================================
# STORAGE LATENCY TESTING
# ============================================

def measure_storage_latency(
    num_iterations: int = 10
) -> Dict[str, Any]:
    """
    Measure storage operation latency
    
    Args:
        num_iterations: Number of iterations
        
    Returns:
        Timing statistics for storage operations
    """
    try:
        from backend.services.storage_service import StorageService
        
        storage = StorageService()
        results = {}
        
        # Test data
        test_data = {
            'test_key': 'test_value',
            'timestamp': '2026-01-14T12:00:00Z',
            'data': ['item1', 'item2', 'item3']
        }
        
        # Measure write operation
        write_times = []
        for i in range(num_iterations):
            with Timer() as t:
                try:
                    storage.save_submission(f'test_benchmark_{i}', test_data)
                except Exception as e:
                    logger.warning(f"Write failed: {e}")
            write_times.append(t.get_elapsed())
        
        results['write'] = compute_timing_stats(write_times)
        
        # Measure read operation
        read_times = []
        for i in range(num_iterations):
            with Timer() as t:
                try:
                    storage.load_submission(f'test_benchmark_{i}')
                except Exception as e:
                    logger.warning(f"Read failed: {e}")
            read_times.append(t.get_elapsed())
        
        results['read'] = compute_timing_stats(read_times)
        
        # Cleanup
        for i in range(num_iterations):
            try:
                storage.delete_submission(f'test_benchmark_{i}')
            except:
                pass
        
        return results
    
    except Exception as e:
        logger.error(f"Failed to measure storage latency: {e}")
        return {'error': str(e)}

# ============================================
# END-TO-END LATENCY TESTING
# ============================================

def measure_end_to_end_latency(
    test_submission: Dict[str, Any],
    num_iterations: int = 10
) -> Dict[str, float]:
    """
    Measure end-to-end submission processing latency
    
    Args:
        test_submission: Test submission data
        num_iterations: Number of iterations
        
    Returns:
        Timing statistics
    """
    try:
        from backend.core.orchestrator import SubmissionOrchestrator
        
        orchestrator = SubmissionOrchestrator()
        times = []
        
        for i in range(num_iterations):
            # Modify submission ID to avoid conflicts
            submission = test_submission.copy()
            submission['submission_id'] = f"benchmark_test_{i}"
            
            with Timer() as t:
                try:
                    orchestrator.process_submission(submission)
                except Exception as e:
                    logger.warning(f"End-to-end processing failed: {e}")
            
            times.append(t.get_elapsed())
        
        return compute_timing_stats(times)
    
    except Exception as e:
        logger.error(f"Failed to measure end-to-end latency: {e}")
        return {'error': str(e)}

# ============================================
# MAIN BENCHMARK FUNCTION
# ============================================

def run_latency_test(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive latency benchmarks
    
    Args:
        config: Benchmark configuration
            - num_iterations: Number of iterations (default: 10)
            - warmup_iterations: Warmup iterations (default: 2)
            - test_layers: List of layers to test (default: all)
            - test_models: List of models to test (default: all)
            - test_storage: Whether to test storage (default: True)
            - test_end_to_end: Whether to test end-to-end (default: True)
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Starting latency benchmarks...")
    
    # Parse configuration
    config = config or {}
    num_iterations = config.get('num_iterations', 10)
    warmup_iterations = config.get('warmup_iterations', 2)
    test_layers = config.get('test_layers', [
        'layer1_anonymity',
        'layer2_credibility',
        'layer3_coordination',
        'layer4_consensus'
    ])
    test_models = config.get('test_models', ['clip'])
    test_storage = config.get('test_storage', True)
    test_end_to_end = config.get('test_end_to_end', True)
    
    # Create benchmark object
    benchmark = LatencyBenchmark(
        num_iterations=num_iterations,
        warmup_iterations=warmup_iterations
    )
    
    results = {
        'configuration': {
            'num_iterations': num_iterations,
            'warmup_iterations': warmup_iterations
        }
    }
    
    # Test storage operations
    if test_storage:
        logger.info("Testing storage operations...")
        try:
            storage_results = measure_storage_latency(num_iterations)
            results['storage'] = storage_results
            logger.info("Storage tests completed")
        except Exception as e:
            logger.error(f"Storage tests failed: {e}")
            results['storage'] = {'error': str(e)}
    
    # Test model inference
    if test_models:
        logger.info("Testing model inference...")
        results['models'] = {}
        
        for model_name in test_models:
            try:
                logger.info(f"Testing {model_name}...")
                
                # Create test input based on model type
                if model_name == 'clip':
                    # Create dummy image
                    import numpy as np
                    test_input = np.random.rand(224, 224, 3).astype(np.float32)
                elif model_name == 'sentence_transformer':
                    test_input = "This is a test sentence for benchmarking."
                else:
                    test_input = None
                
                if test_input is not None:
                    model_results = measure_model_latency(
                        model_name, test_input, num_iterations
                    )
                    results['models'][model_name] = model_results
            except Exception as e:
                logger.error(f"Model {model_name} test failed: {e}")
                results['models'][model_name] = {'error': str(e)}
    
    # Test layer processing
    if test_layers:
        logger.info("Testing layer processing...")
        results['layers'] = {}
        
        for layer_name in test_layers:
            try:
                logger.info(f"Testing {layer_name}...")
                
                # Create test input for layer
                test_input = {
                    'submission_id': 'benchmark_test',
                    'evidence_type': 'image',
                    'narrative': 'Test narrative',
                    'timestamp': '2026-01-14T12:00:00Z'
                }
                
                layer_results = measure_layer_latency(
                    layer_name, test_input, num_iterations
                )
                results['layers'][layer_name] = layer_results
            except Exception as e:
                logger.error(f"Layer {layer_name} test failed: {e}")
                results['layers'][layer_name] = {'error': str(e)}
    
    # Test end-to-end processing
    if test_end_to_end:
        logger.info("Testing end-to-end processing...")
        try:
            test_submission = {
                'submission_id': 'benchmark_e2e',
                'evidence_type': 'text',
                'narrative': 'End-to-end benchmark test',
                'timestamp': '2026-01-14T12:00:00Z'
            }
            
            e2e_results = measure_end_to_end_latency(
                test_submission, num_iterations
            )
            results['end_to_end'] = e2e_results
            logger.info("End-to-end tests completed")
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            results['end_to_end'] = {'error': str(e)}
    
    logger.info("Latency benchmarks completed")
    
    return results

# ============================================
# COMMAND LINE INTERFACE
# ============================================

def main():
    """Command line interface for latency benchmarks"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Run latency benchmarks')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations')
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
        'num_iterations': args.iterations
    }
    
    results = run_latency_test(config)
    
    # Print results
    print("\n" + "="*70)
    print("LATENCY BENCHMARK RESULTS")
    print("="*70)
    
    if 'end_to_end' in results and 'mean' in results['end_to_end']:
        e2e = results['end_to_end']
        print(f"\nEnd-to-End Latency: {e2e['mean']:.3f}s ± {e2e['std']:.3f}s")
        print(f"  Min: {e2e['min']:.3f}s, Max: {e2e['max']:.3f}s")
        print(f"  P95: {e2e['p95']:.3f}s, P99: {e2e['p99']:.3f}s")
    
    if 'layers' in results:
        print("\nLayer Breakdown:")
        for layer, stats in results['layers'].items():
            if 'mean' in stats:
                print(f"  {layer:30s}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    
    if 'models' in results:
        print("\nModel Inference:")
        for model, stats in results['models'].items():
            if 'mean' in stats:
                print(f"  {model:30s}: {stats['mean']:.3f}s ± {stats['std']:.3f}s")
    
    if 'storage' in results:
        print("\nStorage Operations:")
        for op, stats in results['storage'].items():
            if 'mean' in stats:
                print(f"  {op:30s}: {stats['mean']:.6f}s ± {stats['std']:.6f}s")
    
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
    'run_latency_test',
    'measure_layer_latency',
    'measure_model_latency',
    'measure_storage_latency',
    'measure_end_to_end_latency',
    'LatencyBenchmark',
    'Timer',
    'compute_timing_stats'
]

if __name__ == '__main__':
    main()
