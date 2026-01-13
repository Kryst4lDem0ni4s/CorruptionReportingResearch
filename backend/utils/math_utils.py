"""
Math Utils - Statistical calculations and mathematical operations

Provides:
- Bayesian probability calculations
- Confidence interval computation
- Statistical metrics (mean, std, percentiles)
- Entropy calculations
- Normalization functions
- Distance metrics
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

# Initialize logger
logger = logging.getLogger(__name__)


class MathUtils:
    """
    Mathematical utilities for statistical analysis.
    
    Features:
    - Bayesian probability updates
    - Confidence intervals
    - Statistical measures
    - Entropy calculations
    - Normalization methods
    - Distance metrics
    """
    
    @staticmethod
    def bayesian_update(
        prior: float,
        likelihood: float,
        evidence_weight: float = 1.0
    ) -> float:
        """
        Bayesian probability update.
        
        P(H|E) = P(E|H) * P(H) / P(E)
        Simplified: posterior ∝ likelihood * prior
        
        Args:
            prior: Prior probability (0-1)
            likelihood: Likelihood of evidence (0-1)
            evidence_weight: Weight of evidence (default 1.0)
            
        Returns:
            float: Posterior probability (0-1)
        """
        # Weighted likelihood
        weighted_likelihood = likelihood * evidence_weight
        
        # Bayesian update (simplified)
        numerator = weighted_likelihood * prior
        denominator = (weighted_likelihood * prior) + ((1 - weighted_likelihood) * (1 - prior))
        
        if denominator == 0:
            return prior
        
        posterior = numerator / denominator
        
        # Clamp to [0, 1]
        posterior = max(0.0, min(1.0, posterior))
        
        logger.debug(
            f"Bayesian update: prior={prior:.3f}, "
            f"likelihood={likelihood:.3f} → posterior={posterior:.3f}"
        )
        
        return posterior
    
    @staticmethod
    def aggregate_probabilities(
        probabilities: List[float],
        weights: Optional[List[float]] = None,
        method: str = 'weighted_average'
    ) -> float:
        """
        Aggregate multiple probabilities.
        
        Args:
            probabilities: List of probabilities (0-1)
            weights: Optional weights for each probability
            method: Aggregation method ('weighted_average', 'product', 'max', 'min')
            
        Returns:
            float: Aggregated probability (0-1)
        """
        if not probabilities:
            return 0.5  # Neutral
        
        probs = np.array(probabilities)
        
        if weights is None:
            weights = np.ones(len(probs))
        else:
            weights = np.array(weights)
        
        # Normalize weights
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        if method == 'weighted_average':
            result = np.sum(probs * weights)
        
        elif method == 'product':
            # Product rule (assumes independence)
            result = np.prod(probs)
        
        elif method == 'max':
            result = np.max(probs)
        
        elif method == 'min':
            result = np.min(probs)
        
        else:
            # Default to weighted average
            result = np.sum(probs * weights)
        
        return float(result)
    
    @staticmethod
    def calculate_confidence_interval(
        data: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for data.
        
        Args:
            data: List of values
            confidence: Confidence level (0-1)
            
        Returns:
            tuple: (mean, lower_bound, upper_bound)
        """
        if not data:
            return (0.0, 0.0, 0.0)
        
        data_array = np.array(data)
        
        mean = np.mean(data_array)
        std_err = stats.sem(data_array)
        
        # Calculate confidence interval
        interval = stats.t.interval(
            confidence,
            len(data_array) - 1,
            loc=mean,
            scale=std_err
        )
        
        lower_bound = interval[0]
        upper_bound = interval[1]
        
        logger.debug(
            f"Confidence interval ({confidence*100}%): "
            f"{mean:.3f} [{lower_bound:.3f}, {upper_bound:.3f}]"
        )
        
        return (float(mean), float(lower_bound), float(upper_bound))
    
    @staticmethod
    def calculate_entropy(probabilities: List[float]) -> float:
        """
        Calculate Shannon entropy.
        
        H = -Σ p(x) * log2(p(x))
        
        Args:
            probabilities: List of probabilities (must sum to 1)
            
        Returns:
            float: Entropy value
        """
        probs = np.array(probabilities)
        
        # Remove zeros to avoid log(0)
        probs = probs[probs > 0]
        
        if len(probs) == 0:
            return 0.0
        
        # Calculate entropy
        entropy = -np.sum(probs * np.log2(probs))
        
        return float(entropy)
    
    @staticmethod
    def calculate_confidence_score(
        probability: float,
        num_samples: int = 1,
        min_samples: int = 10
    ) -> float:
        """
        Calculate confidence score based on probability and sample size.
        
        High confidence requires both high probability and sufficient samples.
        
        Args:
            probability: Estimated probability
            num_samples: Number of samples
            min_samples: Minimum samples for full confidence
            
        Returns:
            float: Confidence score (0-1)
        """
        # Probability component
        prob_confidence = abs(probability - 0.5) * 2  # Distance from neutral
        
        # Sample size component
        sample_confidence = min(num_samples / min_samples, 1.0)
        
        # Combined confidence
        confidence = prob_confidence * sample_confidence
        
        return float(confidence)
    
    @staticmethod
    def normalize_scores(
        scores: List[float],
        method: str = 'min_max'
    ) -> List[float]:
        """
        Normalize scores to [0, 1] range.
        
        Args:
            scores: List of scores
            method: Normalization method ('min_max', 'z_score', 'softmax')
            
        Returns:
            List of normalized scores
        """
        if not scores:
            return []
        
        scores_array = np.array(scores)
        
        if method == 'min_max':
            # Min-max normalization
            min_val = np.min(scores_array)
            max_val = np.max(scores_array)
            
            if max_val - min_val == 0:
                normalized = np.ones_like(scores_array) * 0.5
            else:
                normalized = (scores_array - min_val) / (max_val - min_val)
        
        elif method == 'z_score':
            # Z-score normalization
            mean = np.mean(scores_array)
            std = np.std(scores_array)
            
            if std == 0:
                normalized = np.ones_like(scores_array) * 0.5
            else:
                z_scores = (scores_array - mean) / std
                # Map to [0, 1] using sigmoid
                normalized = 1 / (1 + np.exp(-z_scores))
        
        elif method == 'softmax':
            # Softmax normalization
            exp_scores = np.exp(scores_array - np.max(scores_array))
            normalized = exp_scores / np.sum(exp_scores)
        
        else:
            # Default to min-max
            min_val = np.min(scores_array)
            max_val = np.max(scores_array)
            
            if max_val - min_val == 0:
                normalized = np.ones_like(scores_array) * 0.5
            else:
                normalized = (scores_array - min_val) / (max_val - min_val)
        
        return normalized.tolist()
    
    @staticmethod
    def calculate_weighted_score(
        scores: Dict[str, float],
        weights: Dict[str, float]
    ) -> float:
        """
        Calculate weighted score from multiple components.
        
        Args:
            scores: Dictionary of component → score
            weights: Dictionary of component → weight
            
        Returns:
            float: Weighted score
        """
        total_score = 0.0
        total_weight = 0.0
        
        for component, score in scores.items():
            weight = weights.get(component, 1.0)
            total_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        weighted_score = total_score / total_weight
        
        return float(weighted_score)
    
    @staticmethod
    def sigmoid(x: float, steepness: float = 1.0) -> float:
        """
        Sigmoid function: 1 / (1 + e^(-steepness * x))
        
        Args:
            x: Input value
            steepness: Steepness of curve
            
        Returns:
            float: Sigmoid output (0-1)
        """
        try:
            result = 1 / (1 + math.exp(-steepness * x))
        except OverflowError:
            # Handle overflow
            result = 0.0 if x < 0 else 1.0
        
        return result
    
    @staticmethod
    def logit(p: float) -> float:
        """
        Logit function: log(p / (1 - p))
        
        Inverse of sigmoid.
        
        Args:
            p: Probability (0-1)
            
        Returns:
            float: Logit value
        """
        # Clamp to avoid division by zero
        p = max(1e-10, min(1 - 1e-10, p))
        
        return math.log(p / (1 - p))
    
    @staticmethod
    def euclidean_distance(
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """
        Calculate Euclidean distance between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Euclidean distance
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        distance = np.linalg.norm(v1 - v2)
        
        return float(distance)
    
    @staticmethod
    def cosine_distance(
        vector1: List[float],
        vector2: List[float]
    ) -> float:
        """
        Calculate cosine distance between two vectors.
        
        Distance = 1 - cosine_similarity
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            float: Cosine distance (0-2)
        """
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        
        cosine_sim = dot_product / (norm1 * norm2)
        
        # Cosine distance
        distance = 1 - cosine_sim
        
        return float(distance)
    
    @staticmethod
    def calculate_percentiles(
        data: List[float],
        percentiles: List[int] = [25, 50, 75]
    ) -> Dict[int, float]:
        """
        Calculate percentiles of data.
        
        Args:
            data: List of values
            percentiles: List of percentile values (0-100)
            
        Returns:
            dict: percentile → value
        """
        if not data:
            return {p: 0.0 for p in percentiles}
        
        data_array = np.array(data)
        
        result = {}
        for p in percentiles:
            value = np.percentile(data_array, p)
            result[p] = float(value)
        
        return result
    
    @staticmethod
    def calculate_statistics(data: List[float]) -> Dict:
        """
        Calculate comprehensive statistics for data.
        
        Args:
            data: List of values
            
        Returns:
            dict: Statistical measures
        """
        if not data:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'q25': 0.0,
                'q75': 0.0
            }
        
        data_array = np.array(data)
        
        stats_dict = {
            'count': len(data_array),
            'mean': float(np.mean(data_array)),
            'median': float(np.median(data_array)),
            'std': float(np.std(data_array)),
            'min': float(np.min(data_array)),
            'max': float(np.max(data_array)),
            'q25': float(np.percentile(data_array, 25)),
            'q75': float(np.percentile(data_array, 75))
        }
        
        return stats_dict
    
    @staticmethod
    def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """
        Clamp value to range.
        
        Args:
            value: Input value
            min_val: Minimum value
            max_val: Maximum value
            
        Returns:
            float: Clamped value
        """
        return max(min_val, min(max_val, value))
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """
        Safe division with default for division by zero.
        
        Args:
            numerator: Numerator
            denominator: Denominator
            default: Default value if denominator is zero
            
        Returns:
            float: Result or default
        """
        if denominator == 0:
            return default
        
        return numerator / denominator
    
    @staticmethod
    def calculate_auroc(
        y_true: List[int],
        y_scores: List[float]
    ) -> float:
        """
        Calculate Area Under ROC Curve.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_scores: Predicted scores (0-1)
            
        Returns:
            float: AUROC score (0-1)
        """
        try:
            from sklearn.metrics import roc_auc_score
            
            auroc = roc_auc_score(y_true, y_scores)
            
            logger.debug(f"AUROC: {auroc:.3f}")
            
            return float(auroc)
            
        except Exception as e:
            logger.error(f"AUROC calculation failed: {e}")
            return 0.5  # Random classifier


# Convenience functions

def bayesian_aggregate(
    prior: float,
    evidences: List[Tuple[float, float]]
) -> float:
    """
    Aggregate multiple evidences using Bayesian updates.
    
    Args:
        prior: Initial prior probability
        evidences: List of (likelihood, weight) tuples
        
    Returns:
        float: Final posterior probability
    """
    posterior = prior
    
    for likelihood, weight in evidences:
        posterior = MathUtils.bayesian_update(posterior, likelihood, weight)
    
    return posterior


def calculate_credibility_score(
    base_score: float,
    num_validators: int,
    agreement_ratio: float
) -> Tuple[float, float]:
    """
    Calculate final credibility score with confidence.
    
    Args:
        base_score: Base credibility score (0-1)
        num_validators: Number of validators
        agreement_ratio: Ratio of validators in agreement (0-1)
        
    Returns:
        tuple: (credibility_score, confidence)
    """
    # Adjust score based on agreement
    credibility = base_score * agreement_ratio
    
    # Calculate confidence based on number of validators
    confidence = MathUtils.calculate_confidence_score(
        credibility,
        num_samples=num_validators,
        min_samples=10
    )
    
    return (credibility, confidence)
