"""
Deepfake Detection Experiment - Real Implementation
Evaluates Layer 2 credibility assessment on deepfake datasets
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import requests
from tqdm import tqdm

from evaluation.datasets.dataset_loader import DatasetLoader
from evaluation.metrics.deepfake_metrics import DeepfakeMetrics
import logging

logger = logging.getLogger(__name__)

class DeepfakeDetectionExperiment:
    """Deepfake detection experiment with real backend processing"""
    
    def __init__(
        self,
        backend_url: str = "http://localhost:8080",
        timeout: int = 300
    ):
        """Initialize experiment"""
        self.backend_url = backend_url
        self.timeout = timeout
        self.metrics_calculator = DeepfakeMetrics()
        
    def run(
        self,
        dataset_name: str = 'faceforensics',
        max_samples: int = 100,
        split: str = 'test'
    ) -> Dict[str, Any]:
        """
        Run deepfake detection experiment
        
        Args:
            dataset_name: Dataset to evaluate
            max_samples: Maximum samples to process
            split: Data split to use
            
        Returns:
            Experiment results with metrics
        """
        logger.info(f"Running deepfake detection on {dataset_name}...")
        
        # 1. Load dataset
        logger.info(f"Loading {dataset_name} dataset (max_samples={max_samples})...")
        try:
            loader = DatasetLoader(dataset_name, verbose=True)
            dataset = loader.load(split=split, max_samples=max_samples, shuffle=True)
            samples = dataset['samples']
            logger.info(f"Loaded {len(samples)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return {'error': str(e), 'status': 'failed'}
        
        # 2. Process samples through backend
        logger.info("Processing samples through backend API...")
        y_true = []
        y_pred = []
        y_score = []
        processing_times = []
        
        for i, sample in enumerate(tqdm(samples, desc="Processing")):
            try:
                # Get ground truth label
                true_label = 1 if sample.get('label') == 'fake' else 0
                y_true.append(true_label)
                
                # Submit to backend
                start_time = time.time()
                result = self._process_sample(sample)
                processing_time = time.time() - start_time
                processing_times.append(processing_time)
                
                # Extract predictions
                credibility_score = result.get('credibility_score', 0.5)
                y_score.append(credibility_score)
                
                # Predict fake if credibility < 0.5 (low credibility = fake)
                predicted_label = 1 if credibility_score < 0.5 else 0
                y_pred.append(predicted_label)
                
                logger.debug(
                    f"Sample {i+1}/{len(samples)}: "
                    f"true={true_label}, pred={predicted_label}, "
                    f"score={credibility_score:.3f}, time={processing_time:.2f}s"
                )
                
            except Exception as e:
                logger.warning(f"Failed to process sample {i} ({sample.get('path')}): {e}")
                # Use neutral predictions for failed samples
                y_score.append(0.5)
                y_pred.append(0)
                processing_times.append(0)  # Use 0 or some default time

        # Filter out None values in y_score just in case
        y_score = [s if s is not None else 0.5 for s in y_score]
        
        # 3. Compute metrics
        logger.info("Computing metrics...")
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_score = np.array(y_score)
        
        metrics_results = self.metrics_calculator.compute_all(
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score
        )
        
        # 4. Prepare results
        summary = self.metrics_calculator.get_summary(metrics_results)
        
        results = {
            'experiment': 'deepfake_detection',
            'dataset': dataset_name,
            'split': split,
            'num_samples': len(samples),
            'num_processed': len(y_pred),
            'metrics': {
                'auroc': summary.get('auroc'),
                'precision': summary.get('precision'),
                'recall': summary.get('recall'),
                'f1_score': summary.get('f1_score'),
                'accuracy': summary.get('accuracy'),
                'balanced_accuracy': summary.get('balanced_accuracy')
            },
            'performance': {
                'avg_processing_time': np.mean(processing_times),
                'total_time': np.sum(processing_times),
                'throughput': len(samples) / np.sum(processing_times)
            },
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_score': y_score.tolist()
            },
            'raw_metrics': {
                name: {
                    'value': result.value,
                    'metadata': result.metadata,
                    'sub_metrics': result.sub_metrics
                }
                for name, result in metrics_results.items()
            }
        }
        
        # Log summary
        logger.info(f"Deepfake Detection Results:")
        logger.info(f"  AUROC: {summary.get('auroc', 0):.4f}")
        logger.info(f"  F1-Score: {summary.get('f1_score', 0):.4f}")
        logger.info(f"  Accuracy: {summary.get('accuracy', 0):.4f}")
        logger.info(f"  Avg Processing Time: {np.mean(processing_times):.2f}s")
        
        return results
    
    def _process_sample(self, sample: Dict) -> Dict[str, Any]:
        """
        Process single sample through backend
        
        Args:
            sample: Sample dictionary with path and metadata
            
        Returns:
            Processing result from backend
        """
        sample_path = Path(sample['path'])
        
        # Prepare submission
        files = {}
        if sample_path.exists():
            # Read file content
            with open(sample_path, 'rb') as f:
                files['file'] = (sample_path.name, f, self._get_content_type(sample_path))
        
        data = {
            'pseudonym': 'evaluation-test',
            'description': f"Evaluation sample: {sample_path.name}",
            'evidence_type': sample.get('type', 'image')
        }
        
        # Submit to backend
        try:
            response = requests.post(
                f"{self.backend_url}/api/v1/submissions",
                files=files if files else None,
                data=data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            submission_id = result.get('submission_id')
            
            # Wait for processing
            return self._wait_for_result(submission_id)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Backend request failed: {e}")
            raise
    
    def _wait_for_result(self, submission_id: str, max_wait: int = 60) -> Dict:
        """
        Wait for submission processing to complete
        
        Args:
            submission_id: Submission ID
            max_wait: Maximum wait time in seconds
            
        Returns:
            Processing result
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"{self.backend_url}/api/v1/submissions/{submission_id}",
                    timeout=10
                )
                response.raise_for_status()
                
                result = response.json()
                status = result.get('status')
                
                if status == 'completed':
                    # Extract credibility score from processing result
                    processing_result = result.get('processing_result', {})
                    credibility = processing_result.get('credibility', {})
                    
                    return {
                        'submission_id': submission_id,
                        'credibility_score': credibility.get('credibility_score', 0.5),
                        'status': 'completed',
                        'result': processing_result
                    }
                elif status == 'failed':
                    logger.warning(f"Submission {submission_id} failed")
                    return {
                        'submission_id': submission_id,
                        'credibility_score': 0.5,
                        'status': 'failed',
                        'error': result.get('error')
                    }
                
                # Still processing, wait
                time.sleep(2)
                
            except Exception as e:
                logger.warning(f"Error checking status: {e}")
                time.sleep(2)
        
        # Timeout
        logger.warning(f"Timeout waiting for {submission_id} after {max_wait} seconds")
        return {
            'submission_id': submission_id,
            'credibility_score': 0.5,
            'status': 'timeout'
        }
    
    def _get_content_type(self, file_path: Path) -> str:
        """Get content type from file extension"""
        ext = file_path.suffix.lower()
        content_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.mp4': 'video/mp4',
            '.avi': 'video/avi'
        }
        return content_types.get(ext, 'application/octet-stream')


def run_experiment(**kwargs) -> Dict[str, Any]:
    """Convenience function to run experiment"""
    experiment = DeepfakeDetectionExperiment()
    return experiment.run(**kwargs)
