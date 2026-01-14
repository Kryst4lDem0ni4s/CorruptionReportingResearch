"""
Layer 4: Byzantine Consensus Simulator
Simulates distributed validators with Byzantine fault tolerance.

Input: Submission with credibility and coordination flags
Output: Consensus decision and vote distribution
"""

import logging
import random
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.services.metrics_service import MetricsService


# Initialize logger
logger = logging.getLogger(__name__)


class DecisionType(str, Enum):
    """Consensus decision types."""
    FORWARD = "forward"  # Forward to authorities
    REVIEW = "review"    # Requires human review
    REJECT = "reject"    # Insufficient evidence


class ValidatorType(str, Enum):
    """Validator behavior types."""
    HONEST = "honest"
    DEVILS_ADVOCATE = "devils_advocate"
    RANDOM = "random"


class Layer4Consensus:
    """
    Layer 4: Byzantine Consensus Simulator
    
    Implements:
    - Simulated validator pool (15-20 validators)
    - Devil's advocate validators (10%)
    - Weighted voting based on validator reputation
    - Adaptive fault tolerance
    - Majority voting with thresholds
    """
    
    def __init__(
        self,
        storage_service,
        num_validators: int = 17,
        devils_advocate_ratio: float = 0.1,
        consensus_threshold: float = 0.67,
        forward_threshold: float = 0.75,
        reject_threshold: float = 0.30,
        metrics_service: Optional[MetricsService] = None
    ):
        """
        Initialize Layer 4 with validator configuration.
        
        Args:
            storage_service: Storage service for validator state
            num_validators: Total number of validators
            devils_advocate_ratio: Ratio of devil's advocate validators
            consensus_threshold: Minimum agreement for consensus
            forward_threshold: Score threshold for forwarding
            reject_threshold: Score threshold for rejection
        """
        self.storage = storage_service
        self.num_validators = num_validators
        self.devils_advocate_ratio = devils_advocate_ratio
        self.consensus_threshold = consensus_threshold
        self.forward_threshold = forward_threshold
        self.reject_threshold = reject_threshold
        self.metrics = metrics_service
        # Initialize or load validators
        self.validators = self._initialize_validators()
        
        logger.info(
            f"Layer 4 (Consensus) initialized with {num_validators} validators "
            f"({int(devils_advocate_ratio*100)}% devil's advocate)"
        )
    
    def process(
        self,
        submission_id: str,
        credibility_score: float,
        coordination_flagged: bool,
        coordination_confidence: float = 0.0
    ) -> Dict:
        """
        Process submission through Byzantine consensus.
        
        Args:
            submission_id: Unique submission identifier
            credibility_score: Final credibility score from Layer 2
            coordination_flagged: Whether coordination was detected
            coordination_confidence: Coordination detection confidence
            
        Returns:
            dict: Consensus results
            
        Raises:
            ValueError: If consensus fails
        """
        logger.info(f"Layer 4 processing submission {submission_id}")
        
        try:
            # Step 1: Adjust score based on coordination flag
            adjusted_score = self._adjust_score_for_coordination(
                credibility_score,
                coordination_flagged,
                coordination_confidence
            )
            
            # Step 2: Get validator votes
            votes = self._collect_votes(
                submission_id,
                adjusted_score,
                coordination_flagged
            )
            
            # Step 3: Aggregate votes with weights
            vote_counts, weighted_scores = self._aggregate_votes(votes)
            
            # Step 4: Determine consensus decision
            decision, agreement = self._determine_decision(
                adjusted_score,
                vote_counts,
                weighted_scores
            )
            
            # Step 5: Update validator reputations
            self._update_validator_reputations(votes, decision)
            
            result = {
                "submission_id": submission_id,
                "decision": decision.value,
                "adjusted_score": adjusted_score,
                "original_score": credibility_score,
                "votes": vote_counts,
                "validator_scores": [v['score'] for v in votes],
                "agreement_percentage": agreement * 100,
                "consensus_reached": agreement >= self.consensus_threshold,
                "num_validators": len(votes),
                "layer4_status": "completed",
                "timestamp_consensus": datetime.utcnow().isoformat()
            }
            
            logger.info(
                f"Layer 4 completed for {submission_id}: "
                f"decision={decision.value}, agreement={agreement:.1%}"
            )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Layer 4 processing failed for {submission_id}: {e}",
                exc_info=True
            )
            raise ValueError(f"Consensus processing failed: {str(e)}")
    
    def _initialize_validators(self) -> List[Dict]:
        """
        Initialize or load validator pool.
        
        Returns:
            list: Validator configurations
        """
        try:
            # Try to load existing validators
            validators = self.storage.load_validators()
            
            if validators and len(validators) == self.num_validators:
                logger.debug(f"Loaded {len(validators)} existing validators")
                return validators
        except Exception:
            pass
        
        # Create new validators
        logger.info(f"Creating {self.num_validators} new validators")
        
        validators = []
        num_devils_advocate = int(self.num_validators * self.devils_advocate_ratio)
        
        for i in range(self.num_validators):
            # Determine validator type
            if i < num_devils_advocate:
                v_type = ValidatorType.DEVILS_ADVOCATE
            else:
                v_type = ValidatorType.HONEST
            
            validator = {
                "id": f"validator_{i+1:02d}",
                "type": v_type.value,
                "reputation": 1.0,  # Start with perfect reputation
                "total_votes": 0,
                "correct_votes": 0,
                "bias": random.uniform(-0.1, 0.1)  # Small random bias
            }
            validators.append(validator)
        
        # Save validators
        try:
            self.storage.save_validators(validators)
        except Exception as e:
            logger.warning(f"Failed to save validators: {e}")
        
        return validators
    
    def _adjust_score_for_coordination(
        self,
        credibility_score: float,
        coordination_flagged: bool,
        coordination_confidence: float
    ) -> float:
        """
        Adjust credibility score if coordination detected.
        
        Args:
            credibility_score: Original credibility score
            coordination_flagged: Whether coordination detected
            coordination_confidence: Detection confidence
            
        Returns:
            float: Adjusted score
        """
        if not coordination_flagged:
            return credibility_score
        
        # Reduce score based on coordination confidence
        penalty = 0.3 * coordination_confidence  # Up to 30% reduction
        adjusted_score = credibility_score * (1.0 - penalty)
        
        logger.debug(
            f"Score adjusted for coordination: "
            f"{credibility_score:.3f} → {adjusted_score:.3f}"
        )
        
        return max(0.0, adjusted_score)
    
    def _collect_votes(
        self,
        submission_id: str,
        score: float,
        coordination_flagged: bool
    ) -> List[Dict]:
        """
        Collect votes from all validators.
        
        Args:
            submission_id: Submission identifier
            score: Adjusted credibility score
            coordination_flagged: Coordination flag
            
        Returns:
            list: Validator votes
        """
        votes = []
        
        for validator in self.validators:
            # Simulate validator's score assessment
            validator_score = self._simulate_validator_vote(
                validator,
                score,
                coordination_flagged
            )
            
            # Determine vote (accept/reject)
            vote = "accept" if validator_score >= 0.5 else "reject"
            
            votes.append({
                "validator_id": validator['id'],
                "validator_type": validator['type'],
                "score": validator_score,
                "vote": vote,
                "weight": validator['reputation']
            })
        
        return votes
    
    def _simulate_validator_vote(
        self,
        validator: Dict,
        score: float,
        coordination_flagged: bool
    ) -> float:
        """
        Simulate how a validator would score the submission.
        
        Args:
            validator: Validator configuration
            score: Credibility score
            coordination_flagged: Coordination flag
            
        Returns:
            float: Validator's score assessment
        """
        validator_type = validator['type']
        bias = validator['bias']
        
        if validator_type == ValidatorType.HONEST.value:
            # Honest validator: small random noise + bias
            noise = random.gauss(0, 0.05)
            validator_score = score + bias + noise
            
        elif validator_type == ValidatorType.DEVILS_ADVOCATE.value:
            # Devil's advocate: systematically skeptical
            # Reduces score, especially for borderline cases
            if 0.4 <= score <= 0.7:
                # Most skeptical in uncertain range
                reduction = random.uniform(0.15, 0.25)
            else:
                reduction = random.uniform(0.05, 0.15)
            
            validator_score = score - reduction
            
        else:  # RANDOM
            # Random validator: uniform random score
            validator_score = random.uniform(0.0, 1.0)
        
        # Additional penalty for coordination-flagged submissions
        if coordination_flagged:
            validator_score *= random.uniform(0.8, 0.95)
        
        # Clip to [0, 1]
        return max(0.0, min(1.0, validator_score))
    
    def _aggregate_votes(
        self,
        votes: List[Dict]
    ) -> Tuple[Dict[str, int], List[float]]:
        """
        Aggregate validator votes with reputation weighting.
        
        Args:
            votes: List of validator votes
            
        Returns:
            tuple: (vote_counts, weighted_scores)
        """
        # Count votes
        vote_counts = {"accept": 0, "reject": 0}
        
        # Weighted scores
        weighted_scores = []
        total_weight = 0.0
        
        for vote in votes:
            vote_counts[vote['vote']] += 1
            
            weight = vote['weight']
            weighted_scores.append(vote['score'] * weight)
            total_weight += weight
        
        # Normalize weighted scores
        if total_weight > 0:
            weighted_scores = [s / total_weight for s in weighted_scores]
        
        return vote_counts, weighted_scores
    
    def _determine_decision(
        self,
        score: float,
        vote_counts: Dict[str, int],
        weighted_scores: List[float]
    ) -> Tuple[DecisionType, float]:
        """
        Determine consensus decision based on votes and thresholds.
        
        Args:
            score: Adjusted credibility score
            vote_counts: Vote counts by type
            weighted_scores: Weighted validator scores
            
        Returns:
            tuple: (decision, agreement_percentage)
        """
        total_votes = sum(vote_counts.values())
        
        if total_votes == 0:
            return DecisionType.REVIEW, 0.0
        
        # Calculate agreement (majority vote percentage)
        max_votes = max(vote_counts.values())
        agreement = max_votes / total_votes
        
        # Determine decision based on score and thresholds
        if score >= self.forward_threshold and vote_counts['accept'] > vote_counts['reject']:
            decision = DecisionType.FORWARD
        elif score <= self.reject_threshold and vote_counts['reject'] > vote_counts['accept']:
            decision = DecisionType.REJECT
        else:
            # Borderline cases require human review
            decision = DecisionType.REVIEW
        
        # Override if consensus not reached
        if agreement < self.consensus_threshold:
            decision = DecisionType.REVIEW
            logger.warning(
                f"Consensus threshold not met (agreement={agreement:.1%}), "
                f"escalating to human review"
            )
        
        return decision, agreement
    
    def _update_validator_reputations(
        self,
        votes: List[Dict],
        decision: DecisionType
    ) -> None:
        """
        Update validator reputations based on consensus outcome.
        
        Validators who voted with the majority gain reputation.
        Devil's advocates maintain reputation for their role.
        
        Args:
            votes: Validator votes
            decision: Final consensus decision
            
        Returns:
            None
        """
        # Determine majority vote
        vote_counts = {"accept": 0, "reject": 0}
        for vote in votes:
            vote_counts[vote['vote']] += 1
        
        majority_vote = "accept" if vote_counts['accept'] > vote_counts['reject'] else "reject"
        
        # Update reputations
        for i, vote in enumerate(votes):
            validator = self.validators[i]
            
            validator['total_votes'] += 1
            
            # Check if validator voted with majority
            voted_with_majority = (vote['vote'] == majority_vote)
            
            if voted_with_majority:
                validator['correct_votes'] += 1
                # Increase reputation slightly
                validator['reputation'] = min(1.0, validator['reputation'] + 0.01)
            else:
                # Decrease reputation slightly (but not for devil's advocates)
                if validator['type'] != ValidatorType.DEVILS_ADVOCATE.value:
                    validator['reputation'] = max(0.1, validator['reputation'] - 0.01)
        
        # Save updated validators
        try:
            self.storage.save_validators(self.validators)
        except Exception as e:
            logger.warning(f"Failed to save validator updates: {e}")
    
    def get_validator_statistics(self) -> Dict:
        """
        Get statistics about validator pool.
        
        Returns:
            dict: Validator statistics
        """
        stats = {
            "total_validators": len(self.validators),
            "avg_reputation": np.mean([v['reputation'] for v in self.validators]),
            "validator_types": {},
            "top_validators": []
        }
        
        # Count by type
        for v in self.validators:
            v_type = v['type']
            stats['validator_types'][v_type] = stats['validator_types'].get(v_type, 0) + 1
        
        # Top 5 validators by reputation
        sorted_validators = sorted(
            self.validators,
            key=lambda v: v['reputation'],
            reverse=True
        )
        
        stats['top_validators'] = [
            {
                "id": v['id'],
                "reputation": v['reputation'],
                "accuracy": v['correct_votes'] / v['total_votes'] if v['total_votes'] > 0 else 0.0
            }
            for v in sorted_validators[:5]
        ]
        
        return stats
