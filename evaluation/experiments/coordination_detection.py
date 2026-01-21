"""
Coordination Detection Experiment - Real Implementation
Evaluates Layer 3 coordination detection on synthetic attack scenarios
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import requests
from datetime import datetime, timedelta

import logging

logger = logging.getLogger(__name__)


class CoordinationDetectionExperiment:
    """Coordination detection experiment with synthetic attacks"""
    
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
        num_scenarios: int = 10,
        submissions_per_scenario: int = 5
    ) -> Dict[str, Any]:
        """
        Run coordination detection experiment
        
        Args:
            num_scenarios: Number of attack scenarios to test
            submissions_per_scenario: Submissions per coordinated group
            
        Returns:
            Experiment results with metrics
        """
        logger.info("Running coordination detection experiment...")
        
        # 1. Generate synthetic coordinated attacks
        logger.info(f"Generating {num_scenarios} attack scenarios...")
        scenarios = self._generate_attack_scenarios(
            num_scenarios,
            submissions_per_scenario
        )
        
        # 2. Submit attacks to backend
        logger.info("Submitting coordinated attacks to backend...")
        y_true = []  # 1 = coordinated, 0 = independent
        y_pred = []
        y_score = []
        
        for i, scenario in enumerate(scenarios):
            logger.info(f"Processing scenario {i+1}/{len(scenarios)}...")
            
            try:
                # Submit all submissions in scenario
                submission_ids = []
                for submission in scenario['submissions']:
                    sub_id = self._submit_to_backend(submission)
                    submission_ids.append(sub_id)
                    time.sleep(1)  # Stagger submissions
                
                # Wait for processing
                time.sleep(5)
                
                # Check coordination detection results
                detection_results = self._check_coordination(submission_ids)
                
                # Ground truth
                is_coordinated = scenario['is_coordinated']
                y_true.append(1 if is_coordinated else 0)
                
                # Prediction from backend
                detected = detection_results.get('is_coordinated', False)
                confidence = detection_results.get('confidence', 0.5)
                
                y_pred.append(1 if detected else 0)
                y_score.append(confidence)
                
                logger.info(
                    f"  Scenario {i+1}: ground_truth={is_coordinated}, "
                    f"detected={detected}, confidence={confidence:.3f}"
                )
                
            except Exception as e:
                logger.warning(f"Failed to process scenario {i+1}: {e}")
                y_true.append(1 if scenario['is_coordinated'] else 0)
                y_pred.append(0)
                y_score.append(0.5)
        
        # 3. Compute metrics
        logger.info("Computing coordination detection metrics...")
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)
        
        metrics = self._compute_metrics(y_true, y_pred, y_score)
        
        # 4. Prepare results
        results = {
            'experiment': 'coordination_detection',
            'num_scenarios': num_scenarios,
            'submissions_per_scenario': submissions_per_scenario,
            'metrics': metrics,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_score': y_score.tolist()
            },
            'scenarios': scenarios
        }
        
        logger.info(f"Coordination Detection Results:")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return results
    
    def _generate_attack_scenarios(
        self,
        num_scenarios: int,
        submissions_per_scenario: int
    ) -> List[Dict]:
        """Generate synthetic coordinated attack scenarios"""
        scenarios = []
        
        for i in range(num_scenarios):
            # Half coordinated, half independent
            is_coordinated = i < num_scenarios // 2
            
            if is_coordinated:
                # Coordinated attack: similar content, timing, style
                scenario = self._generate_coordinated_scenario(
                    submissions_per_scenario,
                    scenario_id=i
                )
            else:
                # Independent submissions: diverse content
                scenario = self._generate_independent_scenario(
                    submissions_per_scenario,
                    scenario_id=i
                )
            
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_coordinated_scenario(
        self,
        num_submissions: int,
        scenario_id: int
    ) -> Dict:
        """Generate coordinated attack with similar patterns"""
        
        # Base narrative template
        base_narrative = f"Officials at location X engaged in misconduct on date Y"
        
        # Similar variations
        narratives = [
            f"Officials at location X engaged in serious misconduct on date Y involving funds",
            f"Government officials at location X were involved in misconduct on date Y with evidence",
            f"Authorities at location X committed misconduct on date Y affecting citizens",
            f"Officials at location X engaged in misconduct on date Y as documented",
            f"Leadership at location X involved in misconduct on date Y with witnesses"
        ][:num_submissions]
        
        # Clustered timestamps (within 1 hour)
        base_time = datetime.now()
        timestamps = [
            (base_time + timedelta(minutes=i*10)).isoformat()
            for i in range(num_submissions)
        ]
        
        submissions = []
        for j in range(num_submissions):
            submissions.append({
                'pseudonym': f'coordinated_user_{scenario_id}_{j}',
                'description': narratives[j % len(narratives)],
                'evidence_type': 'text',
                'timestamp': timestamps[j],
                'metadata': {
                    'scenario_id': scenario_id,
                    'submission_index': j,
                    'coordinated': True
                }
            })
        
        return {
            'scenario_id': scenario_id,
            'is_coordinated': True,
            'num_submissions': num_submissions,
            'submissions': submissions,
            'attack_type': 'coordinated'
        }
    
    def _generate_independent_scenario(
        self,
        num_submissions: int,
        scenario_id: int
    ) -> Dict:
        """Generate independent submissions with diverse patterns"""
        
        # Diverse narratives
        narratives = [
            "Observed infrastructure issues at regional office requiring attention",
            "Healthcare facility experiencing supply shortages affecting patient care",
            "Educational institution facing funding challenges impacting programs",
            "Transportation delays causing disruptions to public services",
            "Environmental concerns raised by community members"
        ][:num_submissions]
        
        # Spread timestamps (over 24 hours)
        base_time = datetime.now()
        timestamps = [
            (base_time + timedelta(hours=i*5)).isoformat()
            for i in range(num_submissions)
        ]
        
        submissions = []
        for j in range(num_submissions):
            submissions.append({
                'pseudonym': f'independent_user_{scenario_id}_{j}',
                'description': narratives[j % len(narratives)],
                'evidence_type': 'text',
                'timestamp': timestamps[j],
                'metadata': {
                    'scenario_id': scenario_id,
                    'submission_index': j,
                    'coordinated': False
                }
            })
        
        return {
            'scenario_id': scenario_id,
            'is_coordinated': False,
            'num_submissions': num_submissions,
            'submissions': submissions,
            'attack_type': 'independent'
        }
    
    def _submit_to_backend(self, submission: Dict) -> str:
        """Submit to backend API"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/submissions",
                json=submission,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get('submission_id')
        except Exception as e:
            logger.error(f"Backend submission failed: {e}")
            raise
    
    def _check_coordination(self, submission_ids: List[str]) -> Dict:
        """Check coordination detection for group of submissions"""
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/coordination/detect",
                json={'submission_ids': submission_ids},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Coordination check failed: {e}")
            return {'is_coordinated': False, 'confidence': 0.5}
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_score: np.ndarray
    ) -> Dict:
        """Compute coordination detection metrics"""
        
        # Confusion matrix
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'precision': round(float(precision), 4),
            'recall': round(float(recall), 4),
            'f1_score': round(float(f1), 4),
            'accuracy': round(float(accuracy), 4),
            'confusion_matrix': {
                'tp': int(tp),
                'fp': int(fp),
                'tn': int(tn),
                'fn': int(fn)
            }
        }


def run_experiment(**kwargs) -> Dict[str, Any]:
    """Convenience function to run experiment"""
    experiment = CoordinationDetectionExperiment()
    return experiment.run(**kwargs)
