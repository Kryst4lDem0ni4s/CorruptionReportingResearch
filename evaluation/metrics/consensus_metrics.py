"""
Corruption Reporting System - Consensus Metrics
Version: 1.0.0
Description: Metrics for evaluating Byzantine consensus simulation

This module provides:
- Convergence time analysis
- Agreement rate metrics
- Fault tolerance evaluation
- Validator behavior analysis

Usage:
    from evaluation.metrics.consensus_metrics import (
        ConvergenceTimeMetric, AgreementRateMetric, ConsensusMetrics
    )
    
    # Compute convergence time
    convergence_metric = ConvergenceTimeMetric()
    result = convergence_metric.compute(validator_votes, rounds)
    
    # Compute all consensus metrics
    metrics = ConsensusMetrics()
    results = metrics.compute_all(validator_history)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np
from collections import Counter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import base metric
try:
    from evaluation.metrics.base_metric import BaseMetric, MetricResult
    from evaluation.metrics import compute_confidence_interval
except ImportError as e:
    print(f"Error importing base metrics: {e}")
    sys.exit(1)

# ============================================
# LOGGING
# ============================================

logger = logging.getLogger('evaluation.metrics.consensus')

# ============================================
# CONVERGENCE TIME METRIC
# ============================================

class ConvergenceTimeMetric(BaseMetric):
    """
    Consensus convergence time metric
    
    Measures how long it takes for validators to reach consensus.
    
    Target: Minimize convergence time while maintaining accuracy
    """
    
    def __init__(self):
        """Initialize convergence time metric"""
        super().__init__(
            name='convergence_time',
            description='Time to reach consensus',
            category='consensus'
        )
    
    def validate_inputs(
        self,
        validator_votes: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """
        Validate inputs
        
        Args:
            validator_votes: List of validator vote dictionaries
            
        Returns:
            True if valid
        """
        if not validator_votes:
            raise ValueError("Validator votes required")
        
        # Check vote structure
        for vote in validator_votes:
            if 'validator_id' not in vote or 'decision' not in vote:
                raise ValueError(
                    f"Vote must contain 'validator_id' and 'decision': {vote}"
                )
        
        return True
    
    def compute(
        self,
        validator_votes: List[Dict[str, Any]],
        rounds: Optional[int] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute convergence time
        
        Args:
            validator_votes: List of validator vote dictionaries
            rounds: Number of consensus rounds (optional)
            
        Returns:
            MetricResult with convergence time
        """
        # Analyze convergence
        convergence_analysis = self._analyze_convergence(validator_votes)
        
        convergence_round = convergence_analysis['convergence_round']
        total_rounds = rounds or convergence_analysis['total_rounds']
        
        # Compute convergence rate
        convergence_rate = convergence_round / total_rounds if total_rounds > 0 else 1.0
        
        return MetricResult(
            name=self.name,
            value=convergence_round,
            metadata={
                'total_rounds': total_rounds,
                'convergence_rate': round(convergence_rate, 4),
                'num_validators': convergence_analysis['num_validators'],
                'final_agreement': convergence_analysis['final_agreement'],
                'converged': convergence_analysis['converged']
            },
            sub_metrics={
                'convergence_round': convergence_round,
                'agreement_threshold': convergence_analysis['agreement_threshold'],
                'max_agreement': convergence_analysis['max_agreement']
            }
        )
    
    def _analyze_convergence(
        self,
        validator_votes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze convergence from validator votes
        
        Args:
            validator_votes: List of validator votes
            
        Returns:
            Convergence analysis dictionary
        """
        # Group votes by round
        votes_by_round = {}
        
        for vote in validator_votes:
            round_num = vote.get('round', 1)
            if round_num not in votes_by_round:
                votes_by_round[round_num] = []
            votes_by_round[round_num].append(vote['decision'])
        
        # Find convergence point
        convergence_round = 1
        converged = False
        max_agreement = 0.0
        final_agreement = 0.0
        
        for round_num in sorted(votes_by_round.keys()):
            decisions = votes_by_round[round_num]
            
            # Compute agreement (percentage of majority vote)
            decision_counts = Counter(decisions)
            majority_count = max(decision_counts.values())
            agreement = majority_count / len(decisions) if decisions else 0
            
            max_agreement = max(max_agreement, agreement)
            
            # Check for convergence (≥80% agreement)
            if agreement >= 0.80 and not converged:
                convergence_round = round_num
                converged = True
            
            # Track final agreement
            final_agreement = agreement
        
        return {
            'convergence_round': convergence_round,
            'total_rounds': max(votes_by_round.keys()) if votes_by_round else 1,
            'num_validators': len(validator_votes),
            'converged': converged,
            'agreement_threshold': 0.80,
            'max_agreement': round(max_agreement, 4),
            'final_agreement': round(final_agreement, 4)
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['validator_votes'],
            'optional': ['rounds'],
            'output_type': 'int'
        }

# ============================================
# AGREEMENT RATE METRIC
# ============================================

class AgreementRateMetric(BaseMetric):
    """
    Validator agreement rate metric
    
    Measures the percentage of validators agreeing on the final decision.
    
    Target: ≥0.80 (80% agreement)
    """
    
    def __init__(self):
        """Initialize agreement rate metric"""
        super().__init__(
            name='agreement_rate',
            description='Validator agreement percentage',
            category='consensus'
        )
    
    def validate_inputs(
        self,
        validator_decisions: List[Any],
        **kwargs
    ) -> bool:
        """Validate inputs"""
        if not validator_decisions:
            raise ValueError("Validator decisions required")
        return True
    
    def compute(
        self,
        validator_decisions: List[Any],
        **kwargs
    ) -> MetricResult:
        """
        Compute agreement rate
        
        Args:
            validator_decisions: List of validator decisions
            
        Returns:
            MetricResult with agreement rate
        """
        # Count decisions
        decision_counts = Counter(validator_decisions)
        
        # Find majority decision
        majority_decision = decision_counts.most_common(1)[0][0]
        majority_count = decision_counts[majority_decision]
        
        # Compute agreement rate
        total_validators = len(validator_decisions)
        agreement_rate = majority_count / total_validators if total_validators > 0 else 0.0
        
        # Check target
        meets_target = agreement_rate >= 0.80
        
        return MetricResult(
            name=self.name,
            value=round(float(agreement_rate), 4),
            metadata={
                'target': 0.80,
                'meets_target': meets_target,
                'num_validators': total_validators,
                'majority_decision': str(majority_decision),
                'majority_count': majority_count,
                'num_unique_decisions': len(decision_counts)
            },
            sub_metrics={
                'agreement_rate': round(float(agreement_rate), 4),
                'disagreement_rate': round(float(1 - agreement_rate), 4),
                'decision_distribution': dict(decision_counts)
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['validator_decisions'],
            'optional': [],
            'output_type': 'float'
        }

# ============================================
# FAULT TOLERANCE METRIC
# ============================================

class FaultToleranceMetric(BaseMetric):
    """
    Byzantine fault tolerance metric
    
    Measures system performance with malicious/faulty validators.
    """
    
    def __init__(self):
        """Initialize fault tolerance metric"""
        super().__init__(
            name='fault_tolerance',
            description='Byzantine fault tolerance',
            category='consensus'
        )
    
    def validate_inputs(
        self,
        validator_decisions: List[Any],
        validator_types: List[str],
        ground_truth: Any,
        **kwargs
    ) -> bool:
        """Validate inputs"""
        if len(validator_decisions) != len(validator_types):
            raise ValueError("Decisions and types must have same length")
        
        if not ground_truth:
            raise ValueError("Ground truth required")
        
        return True
    
    def compute(
        self,
        validator_decisions: List[Any],
        validator_types: List[str],
        ground_truth: Any,
        **kwargs
    ) -> MetricResult:
        """
        Compute fault tolerance
        
        Args:
            validator_decisions: List of validator decisions
            validator_types: List of validator types ('honest', 'byzantine')
            ground_truth: Ground truth decision
            
        Returns:
            MetricResult with fault tolerance analysis
        """
        # Separate honest and byzantine validators
        honest_decisions = [
            dec for dec, vtype in zip(validator_decisions, validator_types)
            if vtype == 'honest'
        ]
        byzantine_decisions = [
            dec for dec, vtype in zip(validator_decisions, validator_types)
            if vtype == 'byzantine'
        ]
        
        # Compute statistics
        num_honest = len(honest_decisions)
        num_byzantine = len(byzantine_decisions)
        total_validators = num_honest + num_byzantine
        
        byzantine_ratio = num_byzantine / total_validators if total_validators > 0 else 0
        
        # Find consensus decision
        decision_counts = Counter(validator_decisions)
        consensus_decision = decision_counts.most_common(1)[0][0] if decision_counts else None
        
        # Check correctness
        correct_consensus = (consensus_decision == ground_truth)
        
        # Compute honest agreement
        honest_correct = sum(1 for dec in honest_decisions if dec == ground_truth)
        honest_accuracy = honest_correct / num_honest if num_honest > 0 else 0
        
        # Compute byzantine impact
        byzantine_correct = sum(1 for dec in byzantine_decisions if dec == ground_truth)
        byzantine_accuracy = byzantine_correct / num_byzantine if num_byzantine > 0 else 0
        
        # Fault tolerance score (1.0 = perfect tolerance)
        fault_tolerance = 1.0 if correct_consensus else 0.0
        
        return MetricResult(
            name=self.name,
            value=round(float(fault_tolerance), 4),
            metadata={
                'num_honest': num_honest,
                'num_byzantine': num_byzantine,
                'byzantine_ratio': round(byzantine_ratio, 4),
                'correct_consensus': correct_consensus,
                'consensus_decision': str(consensus_decision),
                'ground_truth': str(ground_truth)
            },
            sub_metrics={
                'fault_tolerance': round(float(fault_tolerance), 4),
                'honest_accuracy': round(float(honest_accuracy), 4),
                'byzantine_accuracy': round(float(byzantine_accuracy), 4),
                'system_resilience': round(float(fault_tolerance * (1 - byzantine_ratio)), 4)
            }
        )
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['validator_decisions', 'validator_types', 'ground_truth'],
            'optional': [],
            'output_type': 'float'
        }

# ============================================
# VALIDATOR BEHAVIOR METRIC
# ============================================

class ValidatorBehaviorMetric(BaseMetric):
    """
    Validator behavior analysis metric
    
    Analyzes individual validator performance and behavior patterns.
    """
    
    def __init__(self):
        """Initialize validator behavior metric"""
        super().__init__(
            name='validator_behavior',
            description='Validator behavior analysis',
            category='consensus'
        )
    
    def validate_inputs(
        self,
        validator_history: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """Validate inputs"""
        if not validator_history:
            raise ValueError("Validator history required")
        return True
    
    def compute(
        self,
        validator_history: List[Dict[str, Any]],
        ground_truths: Optional[List[Any]] = None,
        **kwargs
    ) -> MetricResult:
        """
        Compute validator behavior metrics
        
        Args:
            validator_history: List of validator vote records
            ground_truths: Ground truth labels (optional)
            
        Returns:
            MetricResult with behavior analysis
        """
        # Organize by validator
        validator_votes = {}
        
        for record in validator_history:
            validator_id = record.get('validator_id')
            if validator_id not in validator_votes:
                validator_votes[validator_id] = []
            validator_votes[validator_id].append(record)
        
        # Analyze each validator
        validator_stats = {}
        
        for validator_id, votes in validator_votes.items():
            stats = self._analyze_validator(votes, ground_truths)
            validator_stats[validator_id] = stats
        
        # Compute aggregate statistics
        avg_accuracy = np.mean([s['accuracy'] for s in validator_stats.values()])
        avg_consistency = np.mean([s['consistency'] for s in validator_stats.values()])
        
        return MetricResult(
            name=self.name,
            value=round(float(avg_accuracy), 4),
            metadata={
                'num_validators': len(validator_stats),
                'total_votes': len(validator_history)
            },
            sub_metrics={
                'avg_accuracy': round(float(avg_accuracy), 4),
                'avg_consistency': round(float(avg_consistency), 4),
                'validator_stats': validator_stats
            }
        )
    
    def _analyze_validator(
        self,
        votes: List[Dict[str, Any]],
        ground_truths: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """Analyze individual validator"""
        num_votes = len(votes)
        
        # Extract decisions
        decisions = [vote.get('decision') for vote in votes]
        
        # Compute consistency (agreement with own majority)
        decision_counts = Counter(decisions)
        majority_decision = decision_counts.most_common(1)[0][0] if decision_counts else None
        consistency = decision_counts[majority_decision] / num_votes if num_votes > 0 else 0
        
        # Compute accuracy if ground truth available
        accuracy = 0.0
        if ground_truths and len(ground_truths) == num_votes:
            correct = sum(1 for dec, truth in zip(decisions, ground_truths) if dec == truth)
            accuracy = correct / num_votes if num_votes > 0 else 0
        
        return {
            'num_votes': num_votes,
            'consistency': round(float(consistency), 4),
            'accuracy': round(float(accuracy), 4),
            'majority_decision': str(majority_decision)
        }
    
    def get_requirements(self) -> Dict[str, Any]:
        """Get metric requirements"""
        return {
            'inputs': ['validator_history'],
            'optional': ['ground_truths'],
            'output_type': 'float'
        }

# ============================================
# COMBINED CONSENSUS METRICS
# ============================================

class ConsensusMetrics:
    """
    Compute all consensus metrics
    
    Convenience class for computing all metrics at once.
    """
    
    def __init__(self):
        """Initialize all metrics"""
        self.convergence_metric = ConvergenceTimeMetric()
        self.agreement_metric = AgreementRateMetric()
        self.fault_tolerance_metric = FaultToleranceMetric()
        self.behavior_metric = ValidatorBehaviorMetric()
        
        logger.info("Initialized ConsensusMetrics")
    
    def compute_all(
        self,
        validator_history: List[Dict[str, Any]],
        ground_truths: Optional[List[Any]] = None,
        validator_types: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        Compute all consensus metrics
        
        Args:
            validator_history: List of validator vote records
            ground_truths: Ground truth labels (optional)
            validator_types: Validator types (optional)
            
        Returns:
            Dictionary of metric results
        """
        results = {}
        
        # Compute convergence time
        try:
            results['convergence_time'] = self.convergence_metric.compute(
                validator_history
            )
        except Exception as e:
            logger.error(f"Failed to compute convergence time: {e}")
        
        # Extract final decisions
        final_decisions = [vote.get('decision') for vote in validator_history]
        
        # Compute agreement rate
        try:
            results['agreement_rate'] = self.agreement_metric.compute(
                final_decisions
            )
        except Exception as e:
            logger.error(f"Failed to compute agreement rate: {e}")
        
        # Compute fault tolerance if validator types provided
        if validator_types and ground_truths:
            try:
                results['fault_tolerance'] = self.fault_tolerance_metric.compute(
                    final_decisions,
                    validator_types,
                    ground_truths[0] if ground_truths else None
                )
            except Exception as e:
                logger.error(f"Failed to compute fault tolerance: {e}")
        
        # Compute validator behavior
        try:
            results['validator_behavior'] = self.behavior_metric.compute(
                validator_history,
                ground_truths
            )
        except Exception as e:
            logger.error(f"Failed to compute validator behavior: {e}")
        
        logger.info(f"Computed {len(results)} metric(s)")
        
        return results

# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def compute_convergence_time(validator_votes: List[Dict[str, Any]]) -> int:
    """
    Compute convergence time
    
    Args:
        validator_votes: Validator vote history
        
    Returns:
        Convergence round
    """
    metric = ConvergenceTimeMetric()
    result = metric.compute(validator_votes)
    return result.value

def compute_agreement_rate(validator_decisions: List[Any]) -> float:
    """
    Compute agreement rate
    
    Args:
        validator_decisions: List of decisions
        
    Returns:
        Agreement rate
    """
    metric = AgreementRateMetric()
    result = metric.compute(validator_decisions)
    return result.value

# ============================================
# PACKAGE EXPORTS
# ============================================

__all__ = [
    'ConvergenceTimeMetric',
    'AgreementRateMetric',
    'FaultToleranceMetric',
    'ValidatorBehaviorMetric',
    'ConsensusMetrics',
    'compute_convergence_time',
    'compute_agreement_rate'
]
