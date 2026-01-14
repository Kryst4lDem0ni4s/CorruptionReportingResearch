"""
Corruption Reporting System - Memory Benchmarks
Version: 1.0.0
Description: Measure memory usage and resource consumption

This module provides:
- Peak memory measurement
- Memory profiling by component
- Memory leak detection
- Resource cleanup verification
- Model memory footprint

Usage:
    from evaluation.benchmarks.memory_test import run_memory_test
    
    results = run_memory_test()
    print(f"Peak Memory: {results['peak_memory_mb']:.2f} MB")
"""

import sys
import os
import time
import logging
import gc
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.benchmarks.memory')

# ============================================
# MEMORY UTILITIES
# ============================================

def get_memory_usage() -> float:
    """
    Get current process memory usage in MB
    
    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert to MB
    except ImportError:
        logger.warning("psutil not available, using fallback method")
        # Fallback: use resource module (Unix only)
        try:
            import resource
            usage = resource.getrusage(resource.RUSAGE_SELF)
            # maxrss is in KB on Linux, bytes on macOS
            if sys.platform == 'darwin':
                return usage.ru_maxrss / (1024 * 1024)
            else:
                return usage.ru_maxrss / 1024
        except:
            return 0.0

def get_system_memory() -> Dict[str, float]:
    """
    Get system memory information
    
    Returns:
        Dictionary with memory stats in MB
    """
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            'total': mem.total / (1024 * 1024),
            'available': mem.available / (1024 * 1024),
            'used': mem.used / (1024 * 1024),
            'percent': mem.percent
        }
    except ImportError:
        logger.warning("psutil not available for system memory stats")
        return {}

def force_garbage_collection():
    """Force garbage collection"""
    gc.collect()
    time.sleep(0.1)  # Allow cleanup

# ============================================
# MEMORY TRACKING
# ============================================

class MemoryTracker:
    """Track memory usage over time"""
    
    def __init__(self):
        self.measurements = []
        self.start_memory = None
        self.baseline_memory = None
    
    def start(self):
        """Start tracking"""
        force_garbage_collection()
        self.start_memory = get_memory_usage()
        self.baseline_memory = self.start_memory
        logger.info(f"Memory tracking started: {self.start_memory:.2f} MB")
    
    def measure(self, label: str = ""):
        """Take memory measurement"""
        memory = get_memory_usage()
        self.measurements.append({
            'label': label,
            'memory_mb': memory,
            'delta_mb': memory - self.baseline_memory,
            'timestamp': time.time()
        })
        return memory
    
    def get_peak(self) -> float:
        """Get peak memory usage"""
        if not self.measurements:
            return 0.0
        return max(m['memory_mb'] for m in self.measurements)
    
    def get_average(self) -> float:
        """Get average memory usage"""
        if not self.measurements:
            return 0.0
        return np.mean([m['memory_mb'] for m in self.measurements])
    
    def get_delta(self) -> float:
        """Get memory delta from baseline"""
        if not self.measurements:
            return 0.0
        current = self.measurements[-1]['memory_mb']
        return current - self.baseline_memory
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        if not self.measurements:
            return {}
        
        memories = [m['memory_mb'] for m in self.measurements]
        deltas = [m['delta_mb'] for m in self.measurements]
        
        return {
            'baseline_mb': self.baseline_memory,
            'peak_mb': max(memories),
            'average_mb': np.mean(memories),
            'final_mb': memories[-1],
            'max_delta_mb': max(deltas),
            'measurements': len(self.measurements),
            'timeline': self.measurements
        }

# ============================================
# COMPONENT MEMORY PROFILING
# ============================================

def measure_model_memory(model_name: str) -> Dict[str, float]:
    """
    Measure memory usage of loading a model
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Memory statistics
    """
    logger.info(f"Measuring memory for model: {model_name}")
    
    tracker = MemoryTracker()
    tracker.start()
    
    try:
        # Load model
        if model_name == 'clip':
            from backend.models.clip_model import CLIPModel
            tracker.measure("before_load")
            model = CLIPModel()
            tracker.measure("after_load")
            
            # Perform inference
            import numpy as np
            test_image = np.random.rand(224, 224, 3).astype(np.float32)
            tracker.measure("before_inference")
            model.predict(test_image)
            tracker.measure("after_inference")
        
        elif model_name == 'wav2vec':
            from backend.models.wav2vec_model import Wav2VecModel
            tracker.measure("before_load")
            model = Wav2VecModel()
            tracker.measure("after_load")
        
        elif model_name == 'blip':
            from backend.models.blip_model import BLIPModel
            tracker.measure("before_load")
            model = BLIPModel()
            tracker.measure("after_load")
        
        elif model_name == 'sentence_transformer':
            from backend.models.sentence_transformer import SentenceTransformerModel
            tracker.measure("before_load")
            model = SentenceTransformerModel()
            tracker.measure("after_load")
            
            # Perform inference
            tracker.measure("before_inference")
            model.encode("Test sentence")
            tracker.measure("after_inference")
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Cleanup
        del model
        force_garbage_collection()
        tracker.measure("after_cleanup")
        
        return tracker.get_summary()
    
    except Exception as e:
        logger.error(f"Failed to measure model {model_name}: {e}")
        return {'error': str(e)}

def measure_layer_memory(layer_name: str) -> Dict[str, float]:
    """
    Measure memory usage of processing a layer
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        Memory statistics
    """
    logger.info(f"Measuring memory for layer: {layer_name}")
    
    tracker = MemoryTracker()
    tracker.start()
    
    try:
        # Create test input
        test_input = {
            'submission_id': 'memory_test',
            'evidence_type': 'text',
            'narrative': 'Memory benchmark test',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
        
        tracker.measure("before_processing")
        
        # Process layer
        if layer_name == 'layer1_anonymity':
            from backend.core.layer1_anonymity import AnonymityLayer
            layer = AnonymityLayer()
            layer.process_submission(test_input)
        
        elif layer_name == 'layer2_credibility':
            from backend.core.layer2_credibility import CredibilityLayer
            layer = CredibilityLayer()
            layer.assess_credibility(test_input)
        
        elif layer_name == 'layer3_coordination':
            from backend.core.layer3_coordination import CoordinationLayer
            layer = CoordinationLayer()
            layer.detect_coordination(test_input)
        
        elif layer_name == 'layer4_consensus':
            from backend.core.layer4_consensus import ConsensusLayer
            layer = ConsensusLayer()
            layer.simulate_consensus(test_input)
        
        else:
            raise ValueError(f"Unknown layer: {layer_name}")
        
        tracker.measure("after_processing")
        
        # Cleanup
        del layer
        force_garbage_collection()
        tracker.measure("after_cleanup")
        
        return tracker.get_summary()
    
    except Exception as e:
        logger.error(f"Failed to measure layer {layer_name}: {e}")
        return {'error': str(e)}

# ============================================
# MEMORY LEAK DETECTION
# ============================================

def detect_memory_leaks(
    num_iterations: int = 100
) -> Dict[str, Any]:
    """
    Detect potential memory leaks through repeated operations
    
    Args:
        num_iterations: Number of iterations
        
    Returns:
        Leak detection results
    """
    logger.info(f"Running memory leak detection: {num_iterations} iterations")
    
    tracker = MemoryTracker()
    tracker.start()
    
    # Measure baseline
    baseline = get_memory_usage()
    measurements = [baseline]
    
    try:
        from backend.core.orchestrator import SubmissionOrchestrator
        orchestrator = SubmissionOrchestrator()
        
        for i in range(num_iterations):
            # Create test submission
            submission = {
                'submission_id': f'leak_test_{i}',
                'evidence_type': 'text',
                'narrative': f'Memory leak test #{i}',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            }
            
            # Process submission
            try:
                orchestrator.process_submission(submission)
            except:
                pass
            
            # Measure memory every 10 iterations
            if i % 10 == 0:
                force_garbage_collection()
                memory = get_memory_usage()
                measurements.append(memory)
                tracker.measure(f"iteration_{i}")
        
        # Final cleanup and measurement
        del orchestrator
        force_garbage_collection()
        final_memory = get_memory_usage()
        measurements.append(final_memory)
        
        # Analyze trend
        memory_delta = final_memory - baseline
        memory_growth_rate = memory_delta / num_iterations if num_iterations > 0 else 0
        
        # Check for leak (threshold: >0.1 MB/iteration)
        has_leak = memory_growth_rate > 0.1
        
        return {
            'baseline_mb': baseline,
            'final_mb': final_memory,
            'delta_mb': memory_delta,
            'growth_rate_mb_per_iter': memory_growth_rate,
            'has_potential_leak': has_leak,
            'iterations': num_iterations,
            'measurements': measurements
        }
    
    except Exception as e:
        logger.error(f"Memory leak detection failed: {e}")
        return {'error': str(e)}

# ============================================
# MAIN BENCHMARK FUNCTION
# ============================================

def run_memory_test(
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run comprehensive memory benchmarks
    
    Args:
        config: Benchmark configuration
            - test_models: List of models to test (default: ['clip'])
            - test_layers: List of layers to test (default: all)
            - test_leaks: Whether to run leak detection (default: True)
            - leak_iterations: Iterations for leak detection (default: 100)
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info("Starting memory benchmarks...")
    
    # Parse configuration
    config = config or {}
    test_models = config.get('test_models', ['clip'])
    test_layers = config.get('test_layers', [
        'layer1_anonymity',
        'layer2_credibility',
        'layer3_coordination',
        'layer4_consensus'
    ])
    test_leaks = config.get('test_leaks', True)
    leak_iterations = config.get('leak_iterations', 100)
    
    results = {
        'configuration': config,
        'system_memory': get_system_memory()
    }
    
    # Test model memory
    if test_models:
        logger.info("Testing model memory...")
        results['models'] = {}
        
        for model_name in test_models:
            try:
                model_results = measure_model_memory(model_name)
                results['models'][model_name] = model_results
            except Exception as e:
                logger.error(f"Model {model_name} test failed: {e}")
                results['models'][model_name] = {'error': str(e)}
    
    # Test layer memory
    if test_layers:
        logger.info("Testing layer memory...")
        results['layers'] = {}
        
        for layer_name in test_layers:
            try:
                layer_results = measure_layer_memory(layer_name)
                results['layers'][layer_name] = layer_results
            except Exception as e:
                logger.error(f"Layer {layer_name} test failed: {e}")
                results['layers'][layer_name] = {'error': str(e)}
    
    # Test memory leaks
    if test_leaks:
        logger.info("Running memory leak detection...")
        try:
            leak_results = detect_memory_leaks(leak_iterations)
            results['leak_detection'] = leak_results
        except Exception as e:
            logger.error(f"Leak detection failed: {e}")
            results['leak_detection'] = {'error': str(e)}
    
    logger.info("Memory benchmarks completed")
    
    return results

# ============================================
# COMMAND LINE INTERFACE
# ============================================

def main():
    """Command line interface for memory benchmarks"""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description='Run memory benchmarks')
    parser.add_argument('--models', nargs='+', default=['clip'],
                        help='Models to test')
    parser.add_argument('--leak-iterations', type=int, default=100,
                        help='Iterations for leak detection')
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
        'test_models': args.models,
        'leak_iterations': args.leak_iterations
    }
    
    results = run_memory_test(config)
    
    # Print results
    print("\n" + "="*70)
    print("MEMORY BENCHMARK RESULTS")
    print("="*70)
    
    if 'system_memory' in results:
        sys_mem = results['system_memory']
        print(f"\nSystem Memory:")
        print(f"  Total:               {sys_mem.get('total', 0):.2f} MB")
        print(f"  Available:           {sys_mem.get('available', 0):.2f} MB")
        print(f"  Used:                {sys_mem.get('used', 0):.2f} MB")
        print(f"  Percent:             {sys_mem.get('percent', 0):.1f}%")
    
    if 'models' in results:
        print("\nModel Memory:")
        for model, stats in results['models'].items():
            if 'peak_mb' in stats:
                print(f"  {model:25s}: Peak {stats['peak_mb']:.2f} MB, Avg {stats['average_mb']:.2f} MB")
    
    if 'layers' in results:
        print("\nLayer Memory:")
        for layer, stats in results['layers'].items():
            if 'peak_mb' in stats:
                print(f"  {layer:25s}: Peak {stats['peak_mb']:.2f} MB, Delta {stats['max_delta_mb']:.2f} MB")
    
    if 'leak_detection' in results:
        leak = results['leak_detection']
        if 'has_potential_leak' in leak:
            print(f"\nMemory Leak Detection:")
            print(f"  Baseline:            {leak['baseline_mb']:.2f} MB")
            print(f"  Final:               {leak['final_mb']:.2f} MB")
            print(f"  Delta:               {leak['delta_mb']:.2f} MB")
            print(f"  Growth Rate:         {leak['growth_rate_mb_per_iter']:.4f} MB/iter")
            print(f"  Potential Leak:      {'YES' if leak['has_potential_leak'] else 'NO'}")
    
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
    'run_memory_test',
    'measure_model_memory',
    'measure_layer_memory',
    'detect_memory_leaks',
    'MemoryTracker',
    'get_memory_usage',
    'get_system_memory'
]

if __name__ == '__main__':
    main()
