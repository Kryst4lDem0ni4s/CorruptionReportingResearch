"""
Orchestrator - Coordinates all 6 layers of the framework

Manages the complete workflow:
1. Evidence Submission: Layer 1 → 2 → 3 → 4
2. Counter-Evidence: Layer 5 (Bayesian aggregation)
3. Report Generation: Layer 6
"""
import traceback
import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from backend.core.layer1_anonymity import Layer1Anonymity
from backend.core.layer2_credibility import Layer2Credibility
from backend.core.layer3_coordination import Layer3Coordination
from backend.core.layer4_consensus import Layer4Consensus
from backend.core.layer5_counter_evidence import Layer5CounterEvidence
from backend.core.layer6_reporting import Layer6Reporting

from backend.services.metrics_service import MetricsService

# Initialize logger
logger = logging.getLogger(__name__)


class ProcessingStatus:
    """Processing status constants."""
    PENDING = "pending"
    PROCESSING = "processing"
    LAYER1_COMPLETE = "layer1_complete"
    LAYER2_COMPLETE = "layer2_complete"
    LAYER3_COMPLETE = "layer3_complete"
    LAYER4_COMPLETE = "layer4_complete"
    LAYER5_COMPLETE = "layer5_complete"
    COMPLETED = "completed"
    FAILED = "failed"


class Orchestrator:
    """
    Orchestrator - Coordinates all 6 layers
    
    Responsibilities:
    - Initialize all layers with required services
    - Process submissions through layers 1-4 sequentially
    - Handle counter-evidence processing (layer 5)
    - Generate forensic reports (layer 6)
    - Manage errors and rollback
    - Track processing metrics
    """
    
    def __init__(
        self,
        storage_service,
        hash_chain_service,
        crypto_service,
        metadata_service,
        validation_service,
        text_utils,
        graph_utils,
        image_utils=None,
        audio_utils=None,
        metrics_service: Optional[MetricsService] = None
    ):
        """
        Initialize orchestrator with all required services.
        
        Args:
            storage_service: Storage service for submissions
            hash_chain_service: Hash chain service for custody
            crypto_service: Cryptography service
            metadata_service: Metadata stripping service
            validation_service: Input validation service
            text_utils: Text utilities for stylometric analysis
            graph_utils: Graph utilities for coordination detection
            image_utils: Image utilities (optional)
            audio_utils: Audio utilities (optional)
        """
        self.storage = storage_service
        self.hash_chain = hash_chain_service
        self.crypto = crypto_service
        self.metadata = metadata_service
        self.validation = validation_service
        self.text_utils = text_utils
        self.graph_utils = graph_utils
        self.image_utils = image_utils
        self.audio_utils = audio_utils
        
        # Initialize all 6 layers
        logger.info("Initializing Orchestrator with 6-layer framework...")
        
        self.layer1 = Layer1Anonymity(
            storage_service=storage_service,
            hash_chain_service=hash_chain_service,
            crypto_service=crypto_service,
            metadata_service=metadata_service,
            metrics_service=metrics_service

        )
        
        self.layer2 = Layer2Credibility(
            storage_service=storage_service,
            validation_service=validation_service,
            image_utils=image_utils,
            audio_utils=audio_utils,
            crypto_service=crypto_service,
            metrics_service=metrics_service
        )
        
        self.layer3 = Layer3Coordination(
            storage_service=storage_service,
            text_utils=text_utils,
            graph_utils=graph_utils,
            metrics_service=metrics_service,
        )
        
        self.layer4 = Layer4Consensus(
            storage_service=storage_service,
            metrics_service=metrics_service

        )
        
        self.layer5 = Layer5CounterEvidence(
            storage_service=storage_service,
            metrics_service=metrics_service

        )
        
        self.layer6 = Layer6Reporting(
            storage_service=storage_service,
            hash_chain_service=hash_chain_service,
            metrics_service=metrics_service

        )
        
        logger.info("Orchestrator initialized successfully with all 6 layers")
    
    async def process_submission(
        self,
        submission_id: str,
        file_path: Path,
        evidence_type: str,
        text_narrative: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process evidence submission through layers 1-4.
        
        Workflow:
        1. Layer 1: Anonymize and secure evidence
        2. Layer 2: Assess credibility (deepfake detection)
        3. Layer 3: Detect coordination patterns
        4. Layer 4: Byzantine consensus
        
        Args:
            submission_id: Unique submission identifier
            file_path: Path to evidence file
            evidence_type: Type of evidence (image/video/audio/document)
            text_narrative: Optional text narrative
            metadata: Optional submission metadata
            
        Returns:
            dict: Complete processing results from all layers
            
        Raises:
            Exception: If processing fails at any layer
        """
        start_time = time.time()
        logger.info(f"Orchestrator processing submission {submission_id}")
        
        # Initialize result dictionary
        result = {
            "id": submission_id,
            "evidence_type": evidence_type,
            "status": ProcessingStatus.PROCESSING,
            "text_narrative": text_narrative,
            "metadata": metadata or {},
            "timestamp_submission": datetime.utcnow().isoformat(),
            "processing_started": datetime.utcnow().isoformat()
        }

        logger.info(f"📋 Debug Info:")
        logger.info(f"  - file_path: {file_path}")
        logger.info(f"  - file_path type: {type(file_path)}")
        logger.info(f"  - file exists: {file_path.exists() if isinstance(file_path, Path) else 'NOT A PATH'}")
        logger.info(f"  - evidence_type: {evidence_type}")
        logger.info(f"  - layer1 initialized: {self.layer1 is not None}")
        logger.info(f"  - crypto service: {self.crypto is not None}")
        logger.info(f"  - hash_chain service: {self.hash_chain is not None}")
        # === END DEBUG BLOCK ===
        
        try:
            # Update status
            self._update_status(submission_id, ProcessingStatus.PROCESSING)
            
            # ===== LAYER 1: ANONYMITY =====
            logger.info(f"[{submission_id}] Starting Layer 1: Anonymity")
            layer1_start = time.time()
            
            layer1_result = await asyncio.to_thread(
                self.layer1.process,
                submission_id=submission_id,
                file_path=file_path,
                evidence_type=evidence_type,
                text_narrative=text_narrative,
                metadata=metadata
            )
            
            layer1_time = time.time() - layer1_start
            result.update(layer1_result)
            result['layer1_time'] = layer1_time
            
            logger.info(
                f"[{submission_id}] Layer 1 complete "
                f"(pseudonym={layer1_result.get('pseudonym')}, time={layer1_time:.2f}s)"
            )
            
            self._update_status(submission_id, ProcessingStatus.LAYER1_COMPLETE)
            self._save_checkpoint(submission_id, result)
            
            # ===== LAYER 2: CREDIBILITY ASSESSMENT =====
            logger.info(f"[{submission_id}] Starting Layer 2: Credibility Assessment")
            layer2_start = time.time()
            
            # Use anonymized/encrypted file path for further analysis if needed
            # Layer 1 returns 'encrypted_file_path'
            anonymized_path = Path(layer1_result.get('encrypted_file_path', file_path))
            
            layer2_result = await asyncio.to_thread(
                self.layer2.process,
                submission_id=submission_id,
                file_path=anonymized_path,
                evidence_type=evidence_type,
                text_narrative=text_narrative
            )
            
            layer2_time = time.time() - layer2_start
            result['credibility'] = layer2_result
            result['layer2_time'] = layer2_time
            
            logger.info(
                f"[{submission_id}] Layer 2 complete "
                f"(score={layer2_result.get('final_score', 0):.3f}, time={layer2_time:.2f}s)"
            )
            
            self._update_status(submission_id, ProcessingStatus.LAYER2_COMPLETE)
            self._save_checkpoint(submission_id, result)
            
            # ===== LAYER 3: COORDINATION DETECTION =====
            logger.info(f"[{submission_id}] Starting Layer 3: Coordination Detection")
            layer3_start = time.time()
            
            layer3_result = await asyncio.to_thread(
                self.layer3.process,
                submission_id=submission_id,
                text_narrative=text_narrative,
                timestamp=datetime.utcnow()
            )
            
            layer3_time = time.time() - layer3_start
            result['coordination'] = layer3_result
            result['layer3_time'] = layer3_time
            
            logger.info(
                f"[{submission_id}] Layer 3 complete "
                f"(flagged={layer3_result.get('flagged')}, time={layer3_time:.2f}s)"
            )
            
            self._update_status(submission_id, ProcessingStatus.LAYER3_COMPLETE)
            self._save_checkpoint(submission_id, result)
            
            # ===== LAYER 4: CONSENSUS =====
            logger.info(f"[{submission_id}] Starting Layer 4: Byzantine Consensus")
            layer4_start = time.time()
            
            layer4_result = await asyncio.to_thread(
                self.layer4.process,
                submission_id=submission_id,
                credibility_score=layer2_result.get('final_score', 0.5),
                coordination_flagged=layer3_result.get('flagged', False),
                coordination_confidence=layer3_result.get('confidence', 0.0)
            )
            
            layer4_time = time.time() - layer4_start
            result['consensus'] = layer4_result
            result['layer4_time'] = layer4_time
            
            logger.info(
                f"[{submission_id}] Layer 4 complete "
                f"(decision={layer4_result.get('decision')}, time={layer4_time:.2f}s)"
            )
            
            self._update_status(submission_id, ProcessingStatus.LAYER4_COMPLETE)
            
            # ===== FINALIZE =====
            total_time = time.time() - start_time
            result['processing_time_seconds'] = total_time
            result['status'] = ProcessingStatus.COMPLETED
            result['timestamp_completed'] = datetime.utcnow().isoformat()
            
            # Save final result
            self.storage.save_submission(submission_id, result)
            self._update_status(submission_id, ProcessingStatus.COMPLETED)
            
            logger.info(
                f"[{submission_id}] Processing completed successfully "
                f"(total_time={total_time:.2f}s)"
            )
            
            return result
            
        # In orchestrator.py, replace the exception handler in process_submission:

        except Exception as e:
            logger.error(
                f"[{submission_id}] Processing failed: {e}",
                exc_info=True
            )

            # Mark as failed - FIRST load existing data
            try:
                existing_data = self.storage.load_submission(submission_id)
                if not existing_data:
                    existing_data = result  # Use what we have so far
                
                # Merge error into existing data
                existing_data.update({
                    'status': ProcessingStatus.FAILED,
                    'error': str(e),
                    'error_traceback': traceback.format_exc(),
                    'timestamp_failed': datetime.utcnow().isoformat()
                })
                
                # Save directly (not update)
                self.storage.save_submission(submission_id, existing_data)
                logger.info(f" Error saved for {submission_id}: {str(e)[:100]}")
                
            except Exception as save_error:
                logger.error(f"❌ CRITICAL: Could not save error: {save_error}")
                logger.error(traceback.format_exc())
            
            self._update_status(submission_id, ProcessingStatus.FAILED)

            # Still raise so routes.py knows it failed
            raise

    
    async def process_counter_evidence(
        self,
        original_submission_id: str,
        counter_evidence_id: str,
        counter_credibility_score: float,
        identity_verified: bool = False
    ) -> Dict:
        """
        Process counter-evidence through Layer 5.
        
        Args:
            original_submission_id: Original submission ID
            counter_evidence_id: Counter-evidence submission ID
            counter_credibility_score: Credibility score of counter-evidence
            identity_verified: Whether defense verified identity
            
        Returns:
            dict: Layer 5 results with updated scores and decision
            
        Raises:
            Exception: If processing fails
        """
        logger.info(
            f"Orchestrator processing counter-evidence: "
            f"{counter_evidence_id} for {original_submission_id}"
        )
        
        try:
            # ===== LAYER 5: COUNTER-EVIDENCE =====
            layer5_start = time.time()
            
            layer5_result = self.layer5.process(
                original_submission_id=original_submission_id,
                counter_evidence_id=counter_evidence_id,
                counter_credibility_score=counter_credibility_score,
                identity_verified=identity_verified
            )
            
            layer5_time = time.time() - layer5_start
            layer5_result['layer5_time'] = layer5_time
            
            logger.info(
                f"Layer 5 complete: {original_submission_id} "
                f"(posterior={layer5_result.get('posterior_score'):.3f}, "
                f"decision_changed={layer5_result.get('decision_changed')})"
            )
            
            # Update original submission with Layer 5 results
            original = self.storage.load_submission(original_submission_id)
            if original:
                original.update({
                    'counter_evidence_id': counter_evidence_id,
                    'posterior_score': layer5_result.get('posterior_score'),
                    'score_delta': layer5_result.get('score_delta'),
                    'new_decision': layer5_result.get('new_decision'),
                    'decision_changed': layer5_result.get('decision_changed'),
                    'identity_verified': identity_verified,
                    'layer5_result': layer5_result,
                    'timestamp_counter_evidence': datetime.utcnow().isoformat()
                })
                
                self.storage.save_submission(original_submission_id, original)
                self._update_status(original_submission_id, ProcessingStatus.LAYER5_COMPLETE)
            
            return layer5_result
            
        except Exception as e:
            logger.error(f"Counter-evidence processing failed: {e}", exc_info=True)
            raise
    
    async def generate_report(
        self,
        submission_id: str,
        include_technical_details: bool = True
    ) -> Path:
        """
        Generate forensic report through Layer 6.
        
        Args:
            submission_id: Submission identifier
            include_technical_details: Include technical analysis details
            
        Returns:
            Path: Path to generated PDF report
            
        Raises:
            Exception: If report generation fails
        """
        logger.info(f"Orchestrator generating report for {submission_id}")
        
        try:
            # ===== LAYER 6: REPORT GENERATION =====
            layer6_start = time.time()
            
            report_path = self.layer6.generate_report(
                submission_id=submission_id,
                include_technical_details=include_technical_details
            )
            
            layer6_time = time.time() - layer6_start
            
            logger.info(
                f"Layer 6 complete: {submission_id} "
                f"(report={report_path.name}, time={layer6_time:.2f}s)"
            )
            
            # Update submission with report info
            submission = self.storage.load_submission(submission_id)
            if submission:
                submission['report_path'] = str(report_path)
                submission['report_generated'] = datetime.utcnow().isoformat()
                submission['layer6_time'] = layer6_time
                self.storage.save_submission(submission_id, submission)
            
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            raise
    
    def get_submission_status(self, submission_id: str) -> Dict:
        """
        Get current processing status of submission.
        
        Args:
            submission_id: Submission identifier
            
        Returns:
            dict: Status information
        """
        try:
            submission = self.storage.load_submission(submission_id)
            
            if not submission:
                return {
                    "id": submission_id,
                    "status": "not_found",
                    "error": "Submission not found"
                }
            
            # Extract key status info
            status_info = {
                "id": submission_id,
                "status": submission.get('status', 'unknown'),
                "pseudonym": submission.get('pseudonym'),
                "evidence_type": submission.get('evidence_type'),
                "timestamp_submission": submission.get('timestamp_submission'),
                "processing_time": submission.get('processing_time_seconds'),
                "layers_completed": self._get_completed_layers(submission),
                "credibility_score": submission.get('credibility', {}).get('final_score'),
                "decision": submission.get('consensus', {}).get('decision'),
                "coordination_flagged": submission.get('coordination', {}).get('flagged'),
                "report_available": 'report_path' in submission
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get submission status: {e}")
            return {
                "id": submission_id,
                "status": "error",
                "error": str(e)
            }
    
    def get_system_statistics(self) -> Dict:
        """
        Get system-wide statistics.
        
        Returns:
            dict: System statistics
        """
        try:
            all_submissions = self.storage.get_all_submissions()
            
            # Calculate statistics
            total_submissions = len(all_submissions)
            completed = sum(1 for s in all_submissions if s.get('status') == ProcessingStatus.COMPLETED)
            failed = sum(1 for s in all_submissions if s.get('status') == ProcessingStatus.FAILED)
            processing = total_submissions - completed - failed
            
            # Credibility statistics
            scores = [
                s.get('credibility', {}).get('final_score', 0)
                for s in all_submissions
                if s.get('credibility')
            ]
            
            # Decision distribution
            decisions = {}
            for s in all_submissions:
                decision = s.get('consensus', {}).get('decision')
                if decision:
                    decisions[decision] = decisions.get(decision, 0) + 1
            
            # Coordination detection
            coordinated = sum(
                1 for s in all_submissions
                if s.get('coordination', {}).get('flagged', False)
            )
            
            # Processing times
            processing_times = [
                s.get('processing_time_seconds', 0)
                for s in all_submissions
                if s.get('processing_time_seconds')
            ]
            
            import numpy as np
            
            stats = {
                "total_submissions": total_submissions,
                "completed": completed,
                "failed": failed,
                "processing": processing,
                "completion_rate": completed / total_submissions if total_submissions > 0 else 0,
                "credibility_scores": {
                    "mean": float(np.mean(scores)) if scores else 0,
                    "median": float(np.median(scores)) if scores else 0,
                    "std": float(np.std(scores)) if scores else 0,
                    "min": float(np.min(scores)) if scores else 0,
                    "max": float(np.max(scores)) if scores else 0
                },
                "decisions": decisions,
                "coordination_detected": coordinated,
                "coordination_rate": coordinated / total_submissions if total_submissions > 0 else 0,
                "processing_times": {
                    "mean": float(np.mean(processing_times)) if processing_times else 0,
                    "median": float(np.median(processing_times)) if processing_times else 0,
                    "min": float(np.min(processing_times)) if processing_times else 0,
                    "max": float(np.max(processing_times)) if processing_times else 0
                },
                "validator_stats": self.layer4.get_validator_statistics(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get system statistics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _update_status(self, submission_id: str, status: str) -> None:
        """
        Update submission status.
        
        Args:
            submission_id: Submission identifier
            status: New status
        """
        try:
            self.storage.update_submission_status(submission_id, status)
        except Exception as e:
            logger.warning(f"Failed to update status for {submission_id}: {e}")
    
    def _save_checkpoint(self, submission_id: str, data: Dict) -> None:
        """
        Save processing checkpoint.
        
        Args:
            submission_id: Submission identifier
            data: Current processing data
        """
        try:
            self.storage.save_submission(submission_id, data)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint for {submission_id}: {e}")
    
    def _get_completed_layers(self, submission: Dict) -> list:
        """
        Get list of completed layers.
        
        Args:
            submission: Submission data
            
        Returns:
            list: Completed layer names
        """
        completed = []
        
        if submission.get('pseudonym'):
            completed.append('layer1_anonymity')
        
        if submission.get('credibility'):
            completed.append('layer2_credibility')
        
        if submission.get('coordination'):
            completed.append('layer3_coordination')
        
        if submission.get('consensus'):
            completed.append('layer4_consensus')
        
        if submission.get('layer5_result'):
            completed.append('layer5_counter_evidence')
        
        if submission.get('report_path'):
            completed.append('layer6_reporting')
        
        return completed
    
    async def retry_failed_submission(
        self,
        submission_id: str,
        retry_from_layer: Optional[int] = None
    ) -> Dict:
        """
        Retry a failed submission.
        
        Args:
            submission_id: Submission identifier
            retry_from_layer: Layer to retry from (1-4), or None for full retry
            
        Returns:
            dict: Processing results
            
        Raises:
            Exception: If retry fails
        """
        logger.info(f"Retrying submission {submission_id} from layer {retry_from_layer}")
        
        try:
            # Load failed submission
            submission = self.storage.load_submission(submission_id)
            if not submission:
                raise ValueError(f"Submission {submission_id} not found")
            
            # Determine what to retry
            if retry_from_layer is None or retry_from_layer == 1:
                # Full retry - extract original file path
                file_path = Path(submission.get('file_path', ''))
                evidence_type = submission.get('evidence_type', 'image')
                text_narrative = submission.get('text_narrative')
                
                return await self.process_submission(
                    submission_id=submission_id,
                    file_path=file_path,
                    evidence_type=evidence_type,
                    text_narrative=text_narrative
                )
            
            else:
                # Partial retry - not implemented for MVP
                logger.warning("Partial retry not supported in MVP")
                raise NotImplementedError("Partial retry not supported in MVP")
                
        except Exception as e:
            logger.error(f"Retry failed for {submission_id}: {e}", exc_info=True)
            raise
    
    def cleanup_old_submissions(self, days: int = 90) -> int:
        """
        Cleanup submissions older than specified days.
        
        Args:
            days: Age threshold in days
            
        Returns:
            int: Number of submissions cleaned up
        """
        logger.info(f"Cleaning up submissions older than {days} days")
        
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            all_submissions = self.storage.get_all_submissions()
            
            cleaned = 0
            for submission in all_submissions:
                timestamp_str = submission.get('timestamp_submission')
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', ''))
                        if timestamp < cutoff_date:
                            # Archive and delete
                            submission_id = submission.get('id')
                            self.storage.archive_submission(submission_id)
                            cleaned += 1
                    except Exception as e:
                        logger.warning(f"Failed to cleanup submission: {e}")
            
            logger.info(f"Cleaned up {cleaned} submissions")
            return cleaned
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def health_check(self) -> Dict:
        """
        Perform health check on all layers.
        
        Returns:
            dict: Health status of each layer
        """
        health = {
            "orchestrator": "healthy",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # Check Layer 1
            health['layer1_anonymity'] = "healthy" if self.layer1 else "unavailable"
            
            # Check Layer 2
            health['layer2_credibility'] = "healthy" if self.layer2 else "unavailable"
            
            # Check Layer 3
            health['layer3_coordination'] = "healthy" if self.layer3 else "unavailable"
            
            # Check Layer 4
            health['layer4_consensus'] = "healthy" if self.layer4 else "unavailable"
            
            # Check Layer 5
            health['layer5_counter_evidence'] = "healthy" if self.layer5 else "unavailable"
            
            # Check Layer 6
            health['layer6_reporting'] = "healthy" if self.layer6 else "unavailable"
            
            # Check storage
            try:
                self.storage.health_check()
                health['storage'] = "healthy"
            except Exception:
                health['storage'] = "unhealthy"
            
            # Check hash chain
            try:
                self.hash_chain.verify_chain()
                health['hash_chain'] = "healthy"
            except Exception:
                health['hash_chain'] = "unhealthy"
            
            # Overall status
            unhealthy = [k for k, v in health.items() if v == "unhealthy"]
            if unhealthy:
                health['overall'] = "degraded"
            else:
                health['overall'] = "healthy"
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health['overall'] = "unhealthy"
            health['error'] = str(e)
        
        return health

    def get_coordination_graph(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        min_similarity: Optional[float] = None
    ) -> Dict:
        """
        Get coordination graph data from Layer 3.
        """
        return self.layer3.get_coordination_graph_data(
            start_date=start_date,
            end_date=end_date,
            min_similarity=min_similarity
        )
