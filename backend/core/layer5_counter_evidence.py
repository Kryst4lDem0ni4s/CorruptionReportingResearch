"""
Layer 5: Counter-Evidence Processor
Bayesian aggregation with presumption-of-innocence weighting.

Input: Original submission + counter-evidence
Output: Updated posterior scores and decision
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import stats

from backend.services.metrics_service import MetricsService


# Initialize logger
logger = logging.getLogger(__name__)


class Layer5CounterEvidence:
    """
    Layer 5: Counter-Evidence Processor
    
    Implements:
    - Bayesian aggregation of accusation and defense
    - 1.3× presumption-of-innocence weighting
    - 1.2× identity verification bonus
    - Posterior probability calculation
    - Decision update logic
    """
    
    def __init__(
        self,
        storage_service,
        presumption_weight: float = 1.3,
        identity_bonus: float = 1.2,
        decision_change_threshold: float = 0.15,
        metrics_service: Optional[MetricsService] = None
    ):
        """
        Initialize Layer 5 with configuration.
        
        Args:
            storage_service: Storage service for submissions
            presumption_weight: Presumption of innocence multiplier
            identity_bonus: Identity verification bonus multiplier
            decision_change_threshold: Minimum score change to update decision
        """
        self.storage = storage_service
        self.presumption_weight = presumption_weight
        self.identity_bonus = identity_bonus
        self.decision_change_threshold = decision_change_threshold
        self.metrics = metrics_service
        logger.info(
            f"Layer 5 (Counter-Evidence) initialized "
            f"(presumption={presumption_weight}×, identity={identity_bonus}×)"
        )
    
    def process(
        self,
        original_submission_id: str,
        counter_evidence_id: str,
        counter_credibility_score: float,
        identity_verified: bool = False
    ) -> Dict:
        """
        Process counter-evidence and update original submission.
        
        Args:
            original_submission_id: Original submission ID
            counter_evidence_id: Counter-evidence submission ID
            counter_credibility_score: Counter-evidence credibility score
            identity_verified: Whether defense verified identity
            
        Returns:
            dict: Aggregation results with updated scores and decision
            
        Raises:
            ValueError: If processing fails
        """
        logger.info(
            f"Layer 5 processing counter-evidence {counter_evidence_id} "
            f"for {original_submission_id}"
        )
        
        try:
            # Step 1: Load original submission
            original = self.storage.load_submission(original_submission_id)
            if not original:
                raise ValueError(f"Original submission {original_submission_id} not found")
            
            # Step 2: Extract original scores
            original_credibility = original.get('credibility', {})
            original_score = original_credibility.get('final_score', 0.5)
            original_decision = original.get('consensus', {}).get('decision', 'review')
            
            logger.debug(
                f"Original: score={original_score:.3f}, decision={original_decision}"
            )
            
            # Step 3: Apply Bayesian aggregation
            posterior_score, likelihood_ratio = self._bayesian_aggregation(
                accusation_score=original_score,
                defense_score=counter_credibility_score,
                identity_verified=identity_verified
            )
            
            # Step 4: Calculate score change
            score_delta = posterior_score - original_score
            
            # Step 5: Determine if decision should change
            decision_changed, new_decision = self._evaluate_decision_change(
                original_score=original_score,
                posterior_score=posterior_score,
                score_delta=score_delta,
                original_decision=original_decision
            )
            
            # Step 6: Calculate confidence intervals
            posterior_ci = self._calculate_posterior_ci(
                original_score=original_score,
                counter_score=counter_credibility_score,
                posterior_score=posterior_score
            )
            
            result = {
                "original_submission_id": original_submission_id,
                "counter_evidence_id": counter_evidence_id,
                "original_score": original_score,
                "counter_score": counter_credibility_score,
                "posterior_score": posterior_score,
                "score_delta": score_delta,
                "likelihood_ratio": likelihood_ratio,
                "identity_verified": identity_verified,
                "identity_bonus_applied": identity_verified,
                "presumption_weight_applied": self.presumption_weight,
                "decision_changed": decision_changed,
                "original_decision": original_decision,
                "new_decision": new_decision,
                "posterior_confidence_interval": posterior_ci,
                "aggregation_method": "bayesian",
                "layer5_status": "completed",
                "timestamp_aggregated": datetime.utcnow().isoformat()
            }
            
            if decision_changed:
                logger.warning(
                    f"Decision changed for {original_submission_id}: "
                    f"{original_decision} → {new_decision} "
                    f"(Δscore={score_delta:+.3f})"
                )
            else:
                logger.info(
                    f"Decision unchanged for {original_submission_id}: "
                    f"{new_decision} (Δscore={score_delta:+.3f})"
                )
            
            return result
            
        except Exception as e:
            logger.error(
                f"Layer 5 processing failed: {e}",
                exc_info=True
            )
            raise ValueError(f"Counter-evidence processing failed: {str(e)}")
    
    def _bayesian_aggregation(
        self,
        accusation_score: float,
        defense_score: float,
        identity_verified: bool
    ) -> Tuple[float, float]:
        """
        Perform Bayesian aggregation of accusation and defense.
        
        Uses Bayes' theorem with presumption-of-innocence weighting:
        P(guilty|evidence) = P(evidence|guilty) * P(guilty) / P(evidence)
        
        Args:
            accusation_score: Original accusation credibility score
            defense_score: Counter-evidence credibility score
            identity_verified: Whether defense verified identity
            
        Returns:
            tuple: (posterior_score, likelihood_ratio)
        """
        # Convert scores to probabilities (already in [0,1])
        p_accusation = accusation_score
        p_defense = defense_score
        
        # Apply identity verification bonus to defense
        if identity_verified:
            p_defense = min(1.0, p_defense * self.identity_bonus)
            logger.debug(
                f"Identity bonus applied: {defense_score:.3f} → {p_defense:.3f}"
            )
        
        # Apply presumption of innocence (weight defense more heavily)
        weighted_defense = p_defense * self.presumption_weight
        
        # Bayesian update: combine accusation and weighted defense
        # Likelihood ratio approach
        
        # P(guilty | accusation)
        prior_guilty = p_accusation
        prior_innocent = 1.0 - p_accusation
        
        # P(defense evidence | guilty) vs P(defense evidence | innocent)
        # If defense score is high, it's more likely the person is innocent
        # Use defense score as P(defense | innocent)
        p_evidence_if_innocent = weighted_defense
        p_evidence_if_guilty = 1.0 - weighted_defense
        
        # Likelihood ratio
        likelihood_ratio = p_evidence_if_innocent / (p_evidence_if_guilty + 1e-10)
        
        # Posterior using Bayes' theorem
        numerator = p_evidence_if_innocent * prior_innocent
        denominator = (p_evidence_if_innocent * prior_innocent + 
                      p_evidence_if_guilty * prior_guilty)
        
        if denominator < 1e-10:
            posterior_innocent = 0.5
        else:
            posterior_innocent = numerator / denominator
        
        # Posterior probability of guilt (what we report)
        posterior_guilty = 1.0 - posterior_innocent
        
        # Clip to [0, 1]
        posterior_score = max(0.0, min(1.0, posterior_guilty))
        
        logger.debug(
            f"Bayesian aggregation: "
            f"P(guilt)={p_accusation:.3f}, "
            f"P(defense|innocent)={weighted_defense:.3f}, "
            f"P(guilt|defense)={posterior_score:.3f}, "
            f"LR={likelihood_ratio:.3f}"
        )
        
        return posterior_score, likelihood_ratio
    
    def _evaluate_decision_change(
        self,
        original_score: float,
        posterior_score: float,
        score_delta: float,
        original_decision: str
    ) -> Tuple[bool, str]:
        """
        Evaluate if decision should change based on posterior score.
        
        Args:
            original_score: Original credibility score
            posterior_score: Posterior score after counter-evidence
            score_delta: Change in score
            original_decision: Original consensus decision
            
        Returns:
            tuple: (decision_changed, new_decision)
        """
        # Decision thresholds (from Layer 4)
        FORWARD_THRESHOLD = 0.75
        REJECT_THRESHOLD = 0.30
        
        # Determine new decision based on posterior score
        if posterior_score >= FORWARD_THRESHOLD:
            new_decision = "forward"
        elif posterior_score <= REJECT_THRESHOLD:
            new_decision = "reject"
        else:
            new_decision = "review"
        
        # Check if decision actually changed
        decision_changed = (new_decision != original_decision)
        
        # Additional check: only change if delta is significant
        if decision_changed and abs(score_delta) < self.decision_change_threshold:
            # Delta too small, keep original decision
            new_decision = original_decision
            decision_changed = False
            logger.debug(
                f"Decision change rejected: delta ({score_delta:.3f}) "
                f"below threshold ({self.decision_change_threshold})"
            )
        
        return decision_changed, new_decision
    
    def _calculate_posterior_ci(
        self,
        original_score: float,
        counter_score: float,
        posterior_score: float,
        confidence: float = 0.90
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval for posterior score.
        
        Args:
            original_score: Original score
            counter_score: Counter-evidence score
            posterior_score: Posterior score
            confidence: Confidence level
            
        Returns:
            tuple: (lower_bound, upper_bound)
        """
        # Estimate variance from score uncertainty
        # Assume both scores have some inherent uncertainty
        
        # Variance estimation: use distance from 0.5 as indicator
        # Scores near 0.5 have higher uncertainty
        original_variance = 0.01 + 0.05 * (1 - 2 * abs(original_score - 0.5))
        counter_variance = 0.01 + 0.05 * (1 - 2 * abs(counter_score - 0.5))
        
        # Combined variance (assuming independence)
        combined_variance = (original_variance + counter_variance) / 2
        std_error = np.sqrt(combined_variance)
        
        # Calculate CI using normal approximation
        z_score = stats.norm.ppf((1 + confidence) / 2)
        margin = z_score * std_error
        
        lower = max(0.0, posterior_score - margin)
        upper = min(1.0, posterior_score + margin)
        
        return (float(lower), float(upper))
    
    def generate_comparison_report(
        self,
        original_submission_id: str,
        counter_evidence_id: str
    ) -> Dict:
        """
        Generate before/after comparison report.
        
        Args:
            original_submission_id: Original submission ID
            counter_evidence_id: Counter-evidence ID
            
        Returns:
            dict: Comparison report data
        """
        try:
            # Load both submissions
            original = self.storage.load_submission(original_submission_id)
            counter = self.storage.load_submission(counter_evidence_id)
            
            if not original or not counter:
                raise ValueError("Submission not found")
            
            # Extract key metrics
            original_cred = original.get('credibility', {})
            counter_cred = counter.get('credibility', {})
            
            comparison = {
                "original": {
                    "submission_id": original_submission_id,
                    "credibility_score": original_cred.get('final_score', 0.0),
                    "decision": original.get('consensus', {}).get('decision', 'unknown'),
                    "coordination_flagged": original.get('coordination', {}).get('flagged', False),
                    "timestamp": original.get('timestamp_submission')
                },
                "counter_evidence": {
                    "submission_id": counter_evidence_id,
                    "credibility_score": counter_cred.get('final_score', 0.0),
                    "identity_verified": counter.get('verified_identity', False),
                    "timestamp": counter.get('timestamp_submission')
                },
                "aggregated": {
                    "posterior_score": original.get('posterior_score'),
                    "score_delta": original.get('score_delta'),
                    "new_decision": original.get('new_decision'),
                    "decision_changed": original.get('decision_changed', False)
                },
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to generate comparison report: {e}")
            return {}
    
    def calculate_impact_metrics(
        self,
        original_score: float,
        posterior_score: float,
        likelihood_ratio: float
    ) -> Dict:
        """
        Calculate impact metrics for research analysis.
        
        Args:
            original_score: Original credibility score
            posterior_score: Posterior score
            likelihood_ratio: Bayesian likelihood ratio
            
        Returns:
            dict: Impact metrics
        """
        score_change_percent = ((posterior_score - original_score) / 
                               (original_score + 1e-10)) * 100
        
        absolute_change = abs(posterior_score - original_score)
        
        # Classify impact level
        if absolute_change < 0.1:
            impact_level = "minimal"
        elif absolute_change < 0.2:
            impact_level = "moderate"
        elif absolute_change < 0.3:
            impact_level = "significant"
        else:
            impact_level = "major"
        
        # Bayes factor interpretation
        if likelihood_ratio > 10:
            evidence_strength = "strong support for innocence"
        elif likelihood_ratio > 3:
            evidence_strength = "moderate support for innocence"
        elif likelihood_ratio > 1:
            evidence_strength = "weak support for innocence"
        elif likelihood_ratio > 0.33:
            evidence_strength = "weak support for guilt"
        elif likelihood_ratio > 0.1:
            evidence_strength = "moderate support for guilt"
        else:
            evidence_strength = "strong support for guilt"
        
        return {
            "score_change_percent": score_change_percent,
            "absolute_change": absolute_change,
            "impact_level": impact_level,
            "likelihood_ratio": likelihood_ratio,
            "evidence_strength": evidence_strength,
            "false_positive_reduction": max(0, original_score - posterior_score) > 0.2
        }

"""from backend.utils.math_utils import MathUtils, bayesian_aggregate
from backend.utils.time_utils import TimeUtils, now

class CounterEvidenceLayer:
    PRESUMPTION_OF_INNOCENCE = 1.3
    IDENTITY_VERIFICATION_BONUS = 1.2
    
    def process_counter_evidence(self, submission_id: str, counter_evidence: dict):
        
        # Get original submission
        original = self.storage.get_submission(submission_id)
        original_score = original.get('credibility_score', 0.5)
        
        # Process counter-evidence
        counter_score = counter_evidence.get('credibility_score', 0.5)
        is_verified = counter_evidence.get('identity_verified', False)
        
        # Apply weights
        counter_weight = self.PRESUMPTION_OF_INNOCENCE
        if is_verified:
            counter_weight *= self.IDENTITY_VERIFICATION_BONUS
        
        # Bayesian update
        prior = original_score
        likelihood = 1 - counter_score  # Inverse for counter-evidence
        
        new_score = MathUtils.bayesian_update(prior, likelihood, counter_weight)
        
        # Calculate confidence
        confidence = MathUtils.calculate_confidence_score(
            new_score,
            num_samples=2,  # Original + counter
            min_samples=3
        )
        
        return {
            'submission_id': submission_id,
            'original_score': original_score,
            'counter_score': counter_score,
            'updated_score': new_score,
            'confidence': confidence,
            'timestamp': now()
        }
"""