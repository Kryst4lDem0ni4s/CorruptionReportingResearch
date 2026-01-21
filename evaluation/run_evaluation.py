#!/usr/bin/env python3
"""
Corruption Reporting System - Evaluation Runner
Version: 1.0.0
Description: Run comprehensive evaluation experiments


This script:
- Runs all evaluation experiments
- Generates performance metrics
- Creates visualizations
- Produces final report


Usage:
    # Run all experiments
    python evaluation/run_evaluation.py
    
    # Run specific experiment
    python evaluation/run_evaluation.py --experiment deepfake_detection
    
    # Run with custom config
    python evaluation/run_evaluation.py --config evaluation/config.yaml
    
    # Generate report only (skip experiments)
    python evaluation/run_evaluation.py --report-only


Experiments:
    1. Deepfake Detection (Layer 2)
    2. Coordination Detection (Layer 3)
    3. Consensus Simulation (Layer 4)
    4. Counter-Evidence Impact (Layer 5)
    5. End-to-End System Performance
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import traceback
from unittest import result

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import after path setup
from backend.logging_config import setup_logging, get_logger
from backend.config import load_config

# Import prometheus_client
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, 
        CollectorRegistry, REGISTRY
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    print("Warning: prometheus_client not available")


# ============================================
# METRICS HELPER
# ============================================

def get_or_create_metric(metric_type, name, description, labelnames=None, registry=None, **kwargs):
    """
    Get existing metric or create new one
    
    Args:
        metric_type: Counter, Gauge, or Histogram class
        name: Metric name
        description: Metric description
        labelnames: Label names (for labeled metrics)
        registry: Prometheus registry (default: REGISTRY)
        **kwargs: Additional metric parameters
        
    Returns:
        Metric instance
    """
    if not HAS_PROMETHEUS:
        return None
    
    registry = registry or REGISTRY
    
    # Try to find existing metric
    for collector in list(registry._collector_to_names.keys()):
        if hasattr(collector, '_name') and collector._name == name:
            return collector
    
    # Create new metric
    try:
        if labelnames:
            return metric_type(name, description, labelnames, registry=registry, **kwargs)
        else:
            return metric_type(name, description, registry=registry, **kwargs)
    except ValueError as e:
        # Metric already exists but couldn't be found - try to use it anyway
        logger = get_logger(__name__)
        logger.warning(f"Could not create/find metric {name}: {e}")
        return None


# ============================================
# PROMETHEUS METRICS (REUSE OR CREATE)
# ============================================

# Application Metrics
submission_total = get_or_create_metric(
    Counter, 
    'submission_total',
    'Total evidence submissions',
    labelnames=['status']
)

submission_processing_duration_seconds = get_or_create_metric(
    Histogram,
    'submission_processing_duration_seconds',
    'Submission processing duration in seconds',
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
)

# Layer-Specific Metrics
layer2_credibility_score_avg = get_or_create_metric(
    Gauge,
    'layer2_credibility_score_avg',
    'Average credibility score'
)

layer2_deepfake_detected_total = get_or_create_metric(
    Counter,
    'layer2_deepfake_detected_total',
    'Total deepfakes detected'
)

layer3_coordination_detected_total = get_or_create_metric(
    Counter,
    'layer3_coordination_detected_total',
    'Total coordinated attacks detected'
)

layer4_consensus_iterations_avg = get_or_create_metric(
    Gauge,
    'layer4_consensus_iterations_avg',
    'Average consensus iterations'
)

layer4_consensus_convergence_time_seconds = get_or_create_metric(
    Gauge,
    'layer4_consensus_convergence_time_seconds',
    'Consensus convergence time in seconds'
)

layer5_counter_evidence_total = get_or_create_metric(
    Counter,
    'layer5_counter_evidence_total',
    'Total counter-evidence submissions'
)

layer5_counter_evidence_impact_percent = get_or_create_metric(
    Gauge,
    'layer5_counter_evidence_impact_percent',
    'Counter-evidence impact percentage'
)

layer6_reports_generated_total = get_or_create_metric(
    Counter,
    'layer6_reports_generated_total',
    'Total reports generated'
)

# Model Metrics
model_inference_duration_seconds = get_or_create_metric(
    Histogram,
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    labelnames=['model'],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)
)

# Evaluation Metrics
evaluation_auroc_score = get_or_create_metric(
    Gauge,
    'evaluation_auroc_score',
    'Current AUROC score from evaluation'
)

evaluation_precision_score = get_or_create_metric(
    Gauge,
    'evaluation_precision_score',
    'Current precision score from evaluation'
)

evaluation_recall_score = get_or_create_metric(
    Gauge,
    'evaluation_recall_score',
    'Current recall score from evaluation'
)


# ============================================
# EXPERIMENT RUNNER
# ============================================

class EvaluationRunner:
    """Main evaluation runner"""
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        verbose: bool = False
    ):
        """
        Initialize evaluation runner
        
        Args:
            config_path: Path to evaluation config
            output_dir: Output directory for results
            verbose: Enable verbose logging
        """
        self.config_path = config_path or (project_root / "evaluation" / "config.yaml")
        self.output_dir = output_dir or (project_root / "evaluation" / "results")
        self.verbose = verbose
        
        # Setup logging
        setup_logging(
            level="DEBUG" if verbose else "INFO",
            console=True,
            file_logging=True
        )
        self.logger = get_logger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results storage
        self.results: Dict[str, Any] = {
            'timestamp': datetime.now().isoformat(),
            'experiments': {},
            'summary': {}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load evaluation configuration"""
        if self.config_path.exists():
            self.logger.info(f"Loading config from: {self.config_path}")
            import yaml
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        else:
            self.logger.warning(f"Config not found: {self.config_path}, using defaults")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default evaluation configuration"""
        return {
            'experiments': {
                'deepfake_detection': {
                    'enabled': True,
                    # 'num_samples': 100,
                    # 'models': ['clip', 'xception']
                },
                'coordination_detection': {
                    'enabled': True,
                    'num_scenarios': 50
                },
                'consensus_simulation': {
                    'enabled': True,
                    'num_validators': 10,
                    # 'num_scenarios': 30
                },
                'counter_evidence': {
                    'enabled': True,
                    'num_cases': 20
                },
                'end_to_end': {
                    'enabled': True,
                    'num_submissions': 50
                }
            },
            'output': {
                'save_visualizations': True,
                'save_detailed_results': True,
                'generate_report': True
            }
        }
    
    def run_all_experiments(self) -> Dict[str, Any]:
        """
        Run all enabled experiments
        
        Returns:
            Dictionary of experiment results
        """
        self.logger.info("="*60)
        self.logger.info("Starting Evaluation Suite")
        self.logger.info("="*60)
        
        experiments_config = self.config.get('experiments', {})
        
        # Run each experiment
        if experiments_config.get('deepfake_detection', {}).get('enabled', True):
            self._run_deepfake_detection()
        
        if experiments_config.get('coordination_detection', {}).get('enabled', True):
            self._run_coordination_detection()
        
        if experiments_config.get('consensus_simulation', {}).get('enabled', True):
            self._run_consensus_simulation()
        
        if experiments_config.get('counter_evidence', {}).get('enabled', True):
            self._run_counter_evidence()
        
        # if experiments_config.get('end_to_end', {}).get('enabled', True):
        #     self._run_end_to_end()
        
        # Generate summary
        self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Generate report
        if self.config.get('output', {}).get('generate_report', True):
            self._generate_report()
        
        self.logger.info("="*60)
        self.logger.info("Evaluation Complete")
        self.logger.info("="*60)
        
        return self.results
    
    def _run_deepfake_detection(self) -> None:
        """Run deepfake detection experiment"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Experiment 1: Deepfake Detection")
        self.logger.info("="*60)
        
        try:
            from evaluation.experiments.deepfake_detection import run_experiment
            
            config = self.config['experiments']['deepfake_detection']
            results = run_experiment(
                # num_samples=config.get('num_samples', 100),
                # models=config.get('models', ['clip', 'xception']),
                # output_dir=self.output_dir / 'deepfake_detection'
            )
            
            self.results['experiments']['deepfake_detection'] = results
            
            # Update metrics
            if evaluation_auroc_score and 'auroc' in results:
                evaluation_auroc_score.set(results['auroc'])
            if evaluation_precision_score and 'precision' in results:
                evaluation_precision_score.set(results['precision'])
            if evaluation_recall_score and 'recall' in results:
                evaluation_recall_score.set(results['recall'])
            
            self.logger.info(f"✓ Deepfake Detection: AUROC={results.get('auroc', 0):.4f}")
            
        except Exception as e:
            self.logger.error(f"✗ Deepfake Detection failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.results['experiments']['deepfake_detection'] = {'error': str(e)}
    
    def _run_coordination_detection(self) -> None:
        """Run coordination detection experiment"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Experiment 2: Coordination Detection")
        self.logger.info("="*60)
        
        try:
            from evaluation.experiments.coordination_detection import run_experiment
            
            config = self.config['experiments']['coordination_detection']
            results = run_experiment(
                # num_scenarios=config.get('num_scenarios', 50),
                # output_dir=self.output_dir / 'coordination_detection'
            )
            
            self.results['experiments']['coordination_detection'] = results
            
            # Update metrics
            if layer3_coordination_detected_total:
                layer3_coordination_detected_total.inc(results.get('total_detected', 0))
            
            self.logger.info(f"✓ Coordination Detection: Accuracy={results.get('accuracy', 0):.4f}")
            
        except Exception as e:
            self.logger.error(f"✗ Coordination Detection failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.results['experiments']['coordination_detection'] = {'error': str(e)}
    
    def _run_consensus_simulation(self) -> None:
        """Run consensus simulation experiment"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Experiment 3: Consensus Simulation")
        self.logger.info("="*60)
        
        try:
            from evaluation.experiments.consensus_simulation import run_experiment
            
            config = self.config['experiments']['consensus_simulation']
            results = run_experiment(
                num_validators=config.get('num_validators', 10),
                # num_scenarios=config.get('num_scenarios', 30),
                # output_dir=self.output_dir / 'consensus_simulation'
            )
            
            self.results['experiments']['consensus_simulation'] = results
            
            # Update metrics
            if layer4_consensus_iterations_avg:
                layer4_consensus_iterations_avg.set(results.get('avg_iterations', 0))
            if layer4_consensus_convergence_time_seconds:
                layer4_consensus_convergence_time_seconds.set(results.get('avg_convergence_time', 0))
            
            self.logger.info(f"✓ Consensus: Convergence={results.get('convergence_rate', 0):.2%}")
            
        except Exception as e:
            self.logger.error(f"✗ Consensus Simulation failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.results['experiments']['consensus_simulation'] = {'error': str(e)}
    
    def _run_counter_evidence(self) -> None:
        """Run counter-evidence experiment"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Experiment 4: Counter-Evidence Impact")
        self.logger.info("="*60)
        
        try:
            from evaluation.experiments.counter_evidence import run_experiment
            
            config = self.config['experiments']['counter_evidence']
            results = run_experiment(
                num_cases=config.get('num_cases', 20),
                # output_dir=self.output_dir / 'counter_evidence'
            )
            
            self.results['experiments']['counter_evidence'] = results
            
            # Update metrics
            if layer5_counter_evidence_total:
                layer5_counter_evidence_total.inc(results.get('total_submissions', 0))
            if layer5_counter_evidence_impact_percent:
                layer5_counter_evidence_impact_percent.set(results.get('avg_impact', 0))
            
            self.logger.info(f"✓ Counter-Evidence: Impact={results.get('avg_impact', 0):.2%}")
            
        except Exception as e:
            self.logger.error(f"✗ Counter-Evidence failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.results['experiments']['counter_evidence'] = {'error': str(e)}
    
    def _run_end_to_end(self) -> None:
        """Run end-to-end system performance experiment"""
        self.logger.info("\n" + "="*60)
        self.logger.info("Experiment 5: End-to-End Performance")
        self.logger.info("="*60)
        
        try:
            from evaluation.experiments.benchmarks import run_experiment
            
            config = self.config['experiments']['end_to_end']
            results = run_experiment(
                num_submissions=config.get('num_submissions', 50),
                output_dir=self.output_dir / 'end_to_end'
            )
            
            self.results['experiments']['end_to_end'] = results
            
            # Update metrics
            if submission_total:
                submission_total.labels(status='completed').inc(results.get('total_submissions', 0))
            
            self.logger.info(f"✓ End-to-End: Success Rate={results.get('success_rate', 0):.2%}")
            
        except Exception as e:
            self.logger.error(f"✗ End-to-End failed: {e}")
            self.logger.debug(traceback.format_exc())
            self.results['experiments']['end_to_end'] = {'error': str(e)}
    
    def _generate_summary(self) -> None:
        """Generate summary statistics"""
        self.logger.info("\nGenerating summary...")
        
        experiments = self.results['experiments']
        
        summary = {
            'total_experiments': len(experiments),
            'successful_experiments': sum(1 for exp in experiments.values() if 'error' not in exp),
            'failed_experiments': sum(1 for exp in experiments.values() if 'error' in exp),
            'key_metrics': {}
        }
        
        # Extract key metrics
        if 'deepfake_detection' in experiments and 'error' not in experiments['deepfake_detection']:
            summary['key_metrics']['deepfake_auroc'] = experiments['deepfake_detection'].get('auroc', 0)
        
        if 'coordination_detection' in experiments and 'error' not in experiments['coordination_detection']:
            summary['key_metrics']['coordination_accuracy'] = experiments['coordination_detection'].get('accuracy', 0)
        
        if 'consensus_simulation' in experiments and 'error' not in experiments['consensus_simulation']:
            summary['key_metrics']['consensus_convergence'] = experiments['consensus_simulation'].get('convergence_rate', 0)
        
        if 'end_to_end' in experiments and 'error' not in experiments['end_to_end']:
            summary['key_metrics']['e2e_success_rate'] = experiments['end_to_end'].get('success_rate', 0)
        
        self.results['summary'] = summary
    
    def _save_results(self) -> None:
        """Save results to JSON"""
        results_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to: {results_file}")
    
    def _generate_report(self) -> None:
        """Generate final evaluation report"""
        self.logger.info("Generating evaluation report...")
        
        try:
            from evaluation.visualizations.plot_roc import generate_all_visualizations
            
            report_path = generate_all_visualizations(
                results_dict=self.results,
                output_dir=self.output_dir
            )
            
            self.logger.info(f"Report generated: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            self.logger.debug(traceback.format_exc())


# ============================================
# COMMAND-LINE INTERFACE
# ============================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Run corruption reporting system evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Path to evaluation config file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['deepfake_detection', 'coordination_detection', 'consensus_simulation', 
                 'counter_evidence', 'end_to_end', 'all'],
        default='all',
        help='Specific experiment to run (default: all)'
    )
    
    parser.add_argument(
        '--report-only',
        action='store_true',
        help='Generate report from existing results'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Create runner
    runner = EvaluationRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    # Run evaluation
    if args.report_only:
        runner.logger.info("Generating report from existing results...")
        runner._generate_report()
    elif args.experiment == 'all':
        runner.run_all_experiments()
    else:
        # Run specific experiment
        runner.logger.info(f"Running experiment: {args.experiment}")
        experiment_method = getattr(runner, f'_run_{args.experiment}', None)
        if experiment_method:
            experiment_method()
            runner._generate_summary()
            runner._save_results()
            if runner.config.get('output', {}).get('generate_report', True):
                runner._generate_report()
        else:
            runner.logger.error(f"Unknown experiment: {args.experiment}")
            return 1
    
    runner.logger.info("\n" + "="*60)
    runner.logger.info("Evaluation Summary")
    runner.logger.info("="*60)
    
    summary = runner.results.get('summary', {})
    runner.logger.info(f"Total Experiments: {summary.get('total_experiments', 0)}")
    runner.logger.info(f"Successful: {summary.get('successful_experiments', 0)}")
    runner.logger.info(f"Failed: {summary.get('failed_experiments', 0)}")
    
    if summary.get('key_metrics'):
        runner.logger.info("\nKey Metrics:")
        for metric, value in summary['key_metrics'].items():
            runner.logger.info(f"  {metric}: {value:.4f}")
    
    runner.logger.info("="*60)
    
    return 0 if summary.get('failed_experiments', 0) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
