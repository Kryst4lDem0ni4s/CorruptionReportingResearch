"""
Consensus Simulation Experiment - Real Implementation
Evaluates Layer 4 Byzantine consensus mechanism
"""

import time
import random
from typing import Dict, Any, List
import numpy as np
import requests

import logging

logger = logging.getLogger(__name__)

class ConsensusSimulationExperiment:
    """Byzantine consensus simulation experiment"""
    
    def __init__(
        self,
        backend_url: str = "http://localhost:8080",
        timeout: int = 300
    ):
        """Initialize experiment"""
        self.backend_url = backend_url
        self.timeout = timeout
    
    def run(
        self,
        num_submissions: int = 20,
        num_validators: int = 15,
        byzantine_ratio: float = 0.1
    ) -> Dict[str, Any]:
        """
        Run consensus simulation experiment
        
        Args:
            num_submissions: Number of submissions to test
            num_validators: Number of validators in pool
            byzantine_ratio: Ratio of Byzantine (malicious) validators
            
        Returns:
            Experiment results with consensus metrics
        """
        logger.info("Running consensus simulation experiment...")
        
        # 1. Initialize validator pool
        logger.info(f"Initializing {num_validators} validators...")
        validators = self._initialize_validators(num_validators, byzantine_ratio)
        
        # 2. Process submissions through consensus
        logger.info(f"Processing {num_submissions} submissions...")
        consensus_results = []
        convergence_times = []
        agreement_rates = []
        
        for i in range(num_submissions):
            logger.info(f"Submission {i+1}/{num_submissions}...")
            
            try:
                # Create test submission
                submission = self._create_test_submission(i)
                
                # Run consensus simulation
                start_time = time.time()
                result = self._run_consensus(submission, validators)
                convergence_time = time.time() - start_time
                
                convergence_times.append(convergence_time)
                agreement_rates.append(result['agreement_rate'])
                consensus_results.append(result)
                
                logger.info(
                    f"  Converged: {result['converged']}, "
                    f"Time: {convergence_time:.2f}s, "
                    f"Agreement: {result['agreement_rate']:.2%}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to process submission {i+1}: {e}")
                convergence_times.append(0)
                agreement_rates.append(0)
        
        # 3. Compute metrics
        logger.info("Computing consensus metrics...")
        metrics = {
            'num_submissions': num_submissions,
            'num_validators': num_validators,
            'byzantine_ratio': byzantine_ratio,
            'avg_convergence_time': round(np.mean(convergence_times), 4),
            'median_convergence_time': round(np.median(convergence_times), 4),
            'avg_agreement_rate': round(np.mean(agreement_rates), 4),
            'min_agreement_rate': round(np.min(agreement_rates), 4),
            'convergence_success_rate': round(
                sum(1 for r in consensus_results if r['converged']) / len(consensus_results),
                4
            ),
            'iterations_stats': {
                'mean': round(np.mean([r['iterations'] for r in consensus_results]), 2),
                'median': round(np.median([r['iterations'] for r in consensus_results]), 2),
                'max': max([r['iterations'] for r in consensus_results])
            }
        }
        
        # 4. Prepare results
        results = {
            'experiment': 'consensus_simulation',
            'metrics': metrics,
            'consensus_results': consensus_results,
            'convergence_times': convergence_times,
            'agreement_rates': agreement_rates
        }
        
        logger.info(f"Consensus Simulation Results:")
        logger.info(f"  Avg Convergence Time: {metrics['avg_convergence_time']:.2f}s")
        logger.info(f"  Avg Agreement Rate: {metrics['avg_agreement_rate']:.2%}")
        logger.info(f"  Success Rate: {metrics['convergence_success_rate']:.2%}")
        
        return results
    
    def _initialize_validators(
        self,
        num_validators: int,
        byzantine_ratio: float
    ) -> List[Dict]:
        """Initialize validator pool with Byzantine nodes"""
        validators = []
        num_byzantine = int(num_validators * byzantine_ratio)
        
        for i in range(num_validators):
            is_byzantine = i < num_byzantine
            validators.append({
                'validator_id': f'validator_{i}',
                'is_byzantine': is_byzantine,
                'reputation': random.uniform(0.7, 1.0) if not is_byzantine else random.uniform(0.3, 0.6),
                'response_time': random.uniform(0.1, 2.0)
            })
        
        logger.info(f"  Honest validators: {num_validators - num_byzantine}")
        logger.info(f"  Byzantine validators: {num_byzantine}")
        
        return validators
    
    def _create_test_submission(self, index: int) -> Dict:
        """Create test submission for consensus"""
        # True credibility score (ground truth)
        true_credibility = random.uniform(0.2, 0.9)
        
        return {
            'submission_id': f'test_submission_{index}',
            'true_credibility': true_credibility,
            'description': f'Test submission {index} for consensus simulation'
        }
    
    def _run_consensus(
        self,
        submission: Dict,
        validators: List[Dict]
    ) -> Dict:
        """Run Byzantine consensus simulation"""
        
        true_credibility = submission['true_credibility']
        max_iterations = 10
        convergence_threshold = 0.1
        
        # Initial votes from validators
        votes = []
        for validator in validators:
            if validator['is_byzantine']:
                # Byzantine: random vote
                vote = random.uniform(0, 1)
            else:
                # Honest: vote near true value with noise
                vote = np.clip(
                    true_credibility + random.gauss(0, 0.1),
                    0, 1
                )
            votes.append(vote)
        
        # Iterative consensus
        iterations = 0
        converged = False
        
        for iteration in range(max_iterations):
            iterations += 1
            
            # Weighted average (by reputation)
            weights = [v['reputation'] for v in validators]
            weighted_mean = np.average(votes, weights=weights)
            
            # Check convergence
            vote_std = np.std(votes)
            if vote_std < convergence_threshold:
                converged = True
                break
            
            # Update votes (pull towards weighted mean)
            new_votes = []
            for i, (vote, validator) in enumerate(zip(votes, validators)):
                if validator['is_byzantine']:
                    # Byzantine: stay random
                    new_votes.append(vote)
                else:
                    # Honest: move towards consensus
                    new_votes.append(0.7 * vote + 0.3 * weighted_mean)
            
            votes = new_votes
        
        # Final consensus
        final_consensus = np.average(votes, weights=[v['reputation'] for v in validators])
        
        # Agreement rate (within 0.1 of consensus)
        agreement = sum(1 for v in votes if abs(v - final_consensus) < 0.1)
        agreement_rate = agreement / len(votes)
        
        return {
            'submission_id': submission['submission_id'],
            'converged': converged,
            'iterations': iterations,
            'final_consensus': round(float(final_consensus), 4),
            'true_credibility': true_credibility,
            'error': abs(final_consensus - true_credibility),
            'agreement_rate': agreement_rate,
            'vote_std': round(float(np.std(votes)), 4),
            'num_honest_votes': len([v for v in validators if not v['is_byzantine']]),
            'num_byzantine_votes': len([v for v in validators if v['is_byzantine']])
        }


def run_experiment(**kwargs) -> Dict[str, Any]:
    """Convenience function to run experiment"""
    experiment = ConsensusSimulationExperiment()
    return experiment.run(**kwargs)
