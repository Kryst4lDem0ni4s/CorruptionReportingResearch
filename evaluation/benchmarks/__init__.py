"""
Corruption Reporting System - Benchmark Package
Version: 1.0.0
Description: Performance testing and profiling suite

This package provides:
- Latency testing (processing time)
- Throughput testing (concurrent load)
- Memory profiling (resource usage)
- Performance reporting

Usage:
    from evaluation.benchmarks import (
        run_latency_test,
        run_throughput_test,
        run_memory_test,
        generate_benchmark_report
    )
    
    # Run benchmarks
    latency_results = run_latency_test()
    throughput_results = run_throughput_test()
    memory_results = run_memory_test()
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.benchmarks')

# ============================================
# IMPORT BENCHMARK MODULES
# ============================================

# Latency testing
try:
    from .latency_test import (
        run_latency_test,
        measure_layer_latency,
        measure_model_latency,
        measure_storage_latency,
        LatencyBenchmark
    )
    _HAS_LATENCY = True
except ImportError as e:
    logger.warning(f"Latency testing unavailable: {e}")
    _HAS_LATENCY = False

# Throughput testing (placeholder for future implementation)
try:
    from .throughput_test import (
        run_throughput_test,
        ThroughputBenchmark
    )
    _HAS_THROUGHPUT = True
except ImportError as e:
    logger.debug(f"Throughput testing not yet implemented: {e}")
    _HAS_THROUGHPUT = False

# Memory testing (placeholder for future implementation)
try:
    from .memory_test import (
        run_memory_test,
        MemoryBenchmark
    )
    _HAS_MEMORY = True
except ImportError as e:
    logger.debug(f"Memory testing not yet implemented: {e}")
    _HAS_MEMORY = False

# ============================================
# BENCHMARK SUITE
# ============================================

def run_all_benchmarks(
    output_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run all available benchmarks
    
    Args:
        output_dir: Directory to save results
        config: Benchmark configuration
        
    Returns:
        Dictionary with all benchmark results
    """
    results = {}
    
    # Latency benchmarks
    if _HAS_LATENCY:
        try:
            logger.info("Running latency benchmarks...")
            latency_results = run_latency_test(config=config)
            results['latency'] = latency_results
            logger.info("Latency benchmarks completed")
        except Exception as e:
            logger.error(f"Latency benchmarks failed: {e}")
            results['latency'] = {'error': str(e)}
    
    # Throughput benchmarks
    if _HAS_THROUGHPUT:
        try:
            logger.info("Running throughput benchmarks...")
            throughput_results = run_throughput_test(config=config)
            results['throughput'] = throughput_results
            logger.info("Throughput benchmarks completed")
        except Exception as e:
            logger.error(f"Throughput benchmarks failed: {e}")
            results['throughput'] = {'error': str(e)}
    
    # Memory benchmarks
    if _HAS_MEMORY:
        try:
            logger.info("Running memory benchmarks...")
            memory_results = run_memory_test(config=config)
            results['memory'] = memory_results
            logger.info("Memory benchmarks completed")
        except Exception as e:
            logger.error(f"Memory benchmarks failed: {e}")
            results['memory'] = {'error': str(e)}
    
    # Save results
    if output_dir:
        save_benchmark_results(results, output_dir)
    
    return results

# ============================================
# RESULT HANDLING
# ============================================

def save_benchmark_results(
    results: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Save benchmark results to JSON file
    
    Args:
        results: Benchmark results
        output_dir: Output directory
        
    Returns:
        Path to saved file
    """
    import json
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }
    
    # Save to file
    filename = f"benchmark_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_path / filename
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Benchmark results saved to: {filepath}")
    
    return str(filepath)

def generate_benchmark_report(
    results: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Generate human-readable benchmark report
    
    Args:
        results: Benchmark results
        output_path: Path to save report (None = return string)
        
    Returns:
        Report text
    """
    lines = []
    lines.append("=" * 70)
    lines.append("BENCHMARK REPORT")
    lines.append("=" * 70)
    lines.append("")
    
    # Latency results
    if 'latency' in results:
        lines.append("LATENCY BENCHMARKS")
        lines.append("-" * 70)
        latency = results['latency']
        
        if 'error' in latency:
            lines.append(f"Error: {latency['error']}")
        else:
            if 'end_to_end' in latency:
                e2e = latency['end_to_end']
                lines.append(f"End-to-End Latency:    {e2e['mean']:.3f}s ± {e2e['std']:.3f}s")
                lines.append(f"  Min: {e2e['min']:.3f}s, Max: {e2e['max']:.3f}s")
            
            if 'layers' in latency:
                lines.append("")
                lines.append("Layer Breakdown:")
                for layer, timing in latency['layers'].items():
                    lines.append(f"  {layer:30s}: {timing['mean']:.3f}s ± {timing['std']:.3f}s")
        
        lines.append("")
    
    # Throughput results
    if 'throughput' in results:
        lines.append("THROUGHPUT BENCHMARKS")
        lines.append("-" * 70)
        throughput = results['throughput']
        
        if 'error' in throughput:
            lines.append(f"Error: {throughput['error']}")
        else:
            if 'requests_per_second' in throughput:
                lines.append(f"Requests/Second:       {throughput['requests_per_second']:.2f}")
            if 'concurrent_users' in throughput:
                lines.append(f"Concurrent Users:      {throughput['concurrent_users']}")
        
        lines.append("")
    
    # Memory results
    if 'memory' in results:
        lines.append("MEMORY BENCHMARKS")
        lines.append("-" * 70)
        memory = results['memory']
        
        if 'error' in memory:
            lines.append(f"Error: {memory['error']}")
        else:
            if 'peak_memory_mb' in memory:
                lines.append(f"Peak Memory Usage:     {memory['peak_memory_mb']:.2f} MB")
            if 'average_memory_mb' in memory:
                lines.append(f"Average Memory Usage:  {memory['average_memory_mb']:.2f} MB")
        
        lines.append("")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Benchmark report saved to: {output_path}")
    
    return report

# ============================================
# UTILITY FUNCTIONS
# ============================================

def check_dependencies() -> Dict[str, bool]:
    """
    Check which benchmark modules are available
    
    Returns:
        Dictionary with availability status
    """
    return {
        'latency': _HAS_LATENCY,
        'throughput': _HAS_THROUGHPUT,
        'memory': _HAS_MEMORY
    }

def print_dependencies():
    """Print dependency status"""
    deps = check_dependencies()
    
    print("\n" + "="*50)
    print("BENCHMARK MODULE DEPENDENCIES")
    print("="*50)
    
    for name, available in deps.items():
        status = "✓ Available" if available else "✗ Not Implemented"
        print(f"{name:25s}: {status}")
    
    print("="*50 + "\n")

# ============================================
# PACKAGE METADATA
# ============================================

__version__ = '1.0.0'
__author__ = 'Corruption Reporting System Team'
__description__ = 'Performance testing and profiling suite'

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    # Configuration
    'run_all_benchmarks',
    'save_benchmark_results',
    'generate_benchmark_report',
    'check_dependencies',
    'print_dependencies',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__'
]

# Add latency testing if available
if _HAS_LATENCY:
    __all__.extend([
        'run_latency_test',
        'measure_layer_latency',
        'measure_model_latency',
        'measure_storage_latency',
        'LatencyBenchmark'
    ])

# Add throughput testing if available
if _HAS_THROUGHPUT:
    __all__.extend([
        'run_throughput_test',
        'ThroughputBenchmark'
    ])

# Add memory testing if available
if _HAS_MEMORY:
    __all__.extend([
        'run_memory_test',
        'MemoryBenchmark'
    ])

# ============================================
# MODULE INITIALIZATION
# ============================================

logger.info(f"Benchmark package v{__version__} initialized")
logger.info(f"Available modules: Latency={_HAS_LATENCY}, Throughput={_HAS_THROUGHPUT}, Memory={_HAS_MEMORY}")
