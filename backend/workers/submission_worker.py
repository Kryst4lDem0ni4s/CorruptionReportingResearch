"""
Submission Worker - Asynchronous submission processing

Handles background processing of evidence submissions through all 6 layers.
Provides non-blocking operation for long-running ML inference.
"""

import asyncio
import logging
import traceback
from pathlib import Path
from typing import Dict, Optional

from backend.core.orchestrator import Orchestrator
from backend.services.storage_service import StorageService
from backend.services.queue_service import QueueService
from backend.services.metrics_service import MetricsService
from backend.utils import get_logger, log_performance, now, TimeUtils

# Initialize logger
logger = get_logger(__name__)


class SubmissionWorker:
    """
    Background worker for processing evidence submissions.
    
    Features:
    - Asynchronous processing through all 6 layers
    - Progress tracking and status updates
    - Error handling and retry logic
    - Graceful shutdown
    - Performance metrics
    """
    
    def __init__(
        self,
        storage_service: StorageService,
        queue_service: QueueService,
        metrics_service: MetricsService,
        orchestrator: Orchestrator,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ):
        """
        Initialize submission worker.
        
        Args:
            storage_service: Storage service instance
            queue_service: Queue service instance
            metrics_service: Metrics service instance
            orchestrator: Orchestrator instance
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries (seconds)
        """
        self.storage = storage_service
        self.queue = queue_service
        self.metrics = metrics_service
        self.orchestrator = orchestrator
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        self.is_running = False
        self.current_task = None
        
        logger.info("SubmissionWorker initialized")
    
    async def start(self):
        """
        Start the worker in the background.
        
        Begins processing submissions from the queue in a non-blocking background task.
        """
        if self.is_running:
            logger.warning("Worker already running")
            return
        
        self.is_running = True
        logger.info("SubmissionWorker starting background task...")
        
        # Create background task for processing loop
        self.worker_task = asyncio.create_task(self._process_loop())
        logger.info("SubmissionWorker background task created")
    
    async def stop(self):
        """
        Stop the worker gracefully.
        
        Cancels the background task and waits for current processing to complete.
        """
        logger.info("Stopping SubmissionWorker...")
        self.is_running = False
        
        # Wait for current task if exists
        if self.current_task:
            try:
                await asyncio.wait_for(self.current_task, timeout=10.0)
            except asyncio.TimeoutError:
                logger.warning("Timed out waiting for current submission to complete")
            except Exception as e:
                logger.error(f"Error waiting for current task: {e}")

        # Cancel main worker loop if it's still running
        if hasattr(self, 'worker_task') and self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                logger.info("Worker task cancelled successfully")
            except Exception as e:
                logger.error(f"Error cancelling worker task: {e}")
        
        logger.info("SubmissionWorker stopped")
    
    async def _process_loop(self):
        """
        Main processing loop.
        
        Continuously polls queue and processes submissions.
        """
        while self.is_running:
            try:
                # Get next submission from queue
                submission_id = self.queue.dequeue('submissions')
                
                if submission_id:
                    # Process submission
                    self.current_task = asyncio.create_task(
                        self.process_submission(submission_id)
                    )
                    await self.current_task
                    self.current_task = None
                else:
                    # No submissions, wait briefly
                    await asyncio.sleep(1.0)
            
            except Exception as e:
                logger.error(f"Error in processing loop: {e}", exc_info=True)
                await asyncio.sleep(5.0)
    
    async def process_submission(
        self,
        submission_id: str,
        retry_count: int = 0
    ) -> Dict:
        """
        Process a single submission through all layers.
        
        Args:
            submission_id: Submission ID
            retry_count: Current retry attempt
            
        Returns:
            dict: Processing result
        """
        start_time = now()
        
        logger.info(f"Processing submission {submission_id} (attempt {retry_count + 1})")
        
        try:
            # Update status to processing
            self._update_status(submission_id, 'processing', progress=0)
            
            # Load submission data
            submission = self.storage.get_submission(submission_id)
            
            if not submission:
                raise ValueError(f"Submission {submission_id} not found")
            
            # Process through orchestrator
            with log_performance(logger, f"Full pipeline for {submission_id}"):
                result = await self._process_through_layers(submission_id, submission)
            
            # Update final status
            self._update_status(
                submission_id,
                'completed',
                progress=100,
                result=result
            )
            
            # Record metrics
            duration = now() - start_time
            self.metrics.record_processing_time(duration)
            self.metrics.increment_counter('submissions_processed')
            
            logger.info(
                f"Submission {submission_id} completed in {duration:.2f}s - "
                f"Score: {result.get('final_credibility_score', 0):.2f}"
            )
            
            return result
        
        except Exception as e:
            logger.error(
                f"Error processing submission {submission_id}: {e}",
                exc_info=True
            )
            
            # Retry logic
            if retry_count < self.max_retries:
                logger.info(
                    f"Retrying submission {submission_id} "
                    f"(attempt {retry_count + 2}/{self.max_retries + 1})"
                )
                
                await asyncio.sleep(self.retry_delay)
                
                return await self.process_submission(
                    submission_id,
                    retry_count=retry_count + 1
                )
            else:
                # Max retries exceeded
                logger.error(
                    f"Max retries exceeded for submission {submission_id}"
                )
                
                self._update_status(
                    submission_id,
                    'failed',
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                
                self.metrics.increment_counter('submissions_failed')
                
                raise
    
    async def _process_through_layers(
        self,
        submission_id: str,
        submission: Dict
    ) -> Dict:
        """
        Process submission through all 6 layers.
        
        Args:
            submission_id: Submission ID
            submission: Submission data
            
        Returns:
            dict: Processing result
        """
        # Layer 1: Anonymity (already applied during submission)
        self._update_status(submission_id, 'processing', progress=10)
        logger.debug(f"{submission_id}: Layer 1 (Anonymity) - Already applied")
        
        # Layer 2: Credibility Assessment
        self._update_status(submission_id, 'processing', progress=20)
        logger.debug(f"{submission_id}: Layer 2 (Credibility) - Starting")
        
        credibility_result = await asyncio.to_thread(
            self.orchestrator.layer2.analyze_credibility,
            submission_id,
            submission
        )
        
        self._update_status(submission_id, 'processing', progress=40)
        logger.debug(
            f"{submission_id}: Layer 2 completed - "
            f"Score: {credibility_result.get('credibility_score', 0):.2f}"
        )
        
        # Layer 3: Coordination Detection
        self._update_status(submission_id, 'processing', progress=50)
        logger.debug(f"{submission_id}: Layer 3 (Coordination) - Starting")
        
        coordination_result = await asyncio.to_thread(
            self.orchestrator.layer3.detect_coordination,
            submission_id
        )
        
        self._update_status(submission_id, 'processing', progress=60)
        logger.debug(
            f"{submission_id}: Layer 3 completed - "
            f"Suspicious: {coordination_result.get('is_coordinated', False)}"
        )
        
        # Layer 4: Consensus
        self._update_status(submission_id, 'processing', progress=70)
        logger.debug(f"{submission_id}: Layer 4 (Consensus) - Starting")
        
        consensus_result = await asyncio.to_thread(
            self.orchestrator.layer4.reach_consensus,
            submission_id,
            credibility_result,
            coordination_result
        )
        
        self._update_status(submission_id, 'processing', progress=80)
        logger.debug(
            f"{submission_id}: Layer 4 completed - "
            f"Consensus: {consensus_result.get('consensus_reached', False)}"
        )
        
        # Layer 5: Counter-Evidence (check if exists)
        self._update_status(submission_id, 'processing', progress=85)
        logger.debug(f"{submission_id}: Layer 5 (Counter-Evidence) - Checking")
        
        counter_evidence = submission.get('counter_evidence', [])
        
        if counter_evidence:
            counter_result = await asyncio.to_thread(
                self.orchestrator.layer5.process_counter_evidence,
                submission_id,
                counter_evidence
            )
            logger.debug(
                f"{submission_id}: Counter-evidence processed - "
                f"Updated score: {counter_result.get('updated_score', 0):.2f}"
            )
        else:
            counter_result = None
            logger.debug(f"{submission_id}: No counter-evidence")
        
        # Layer 6: Report Generation (deferred until requested)
        self._update_status(submission_id, 'processing', progress=90)
        logger.debug(f"{submission_id}: Layer 6 (Reporting) - Deferred")
        
        # Aggregate results
        final_result = {
            'submission_id': submission_id,
            'credibility': credibility_result,
            'coordination': coordination_result,
            'consensus': consensus_result,
            'counter_evidence': counter_result,
            'final_credibility_score': consensus_result.get('final_score', 0.5),
            'processing_timestamp': now(),
            'status': 'completed'
        }
        
        # Save result
        self.storage.update_submission(submission_id, {
            'processing_result': final_result,
            'status': 'completed',
            'completed_at': now()
        })
        
        return final_result
    
    def _update_status(
        self,
        submission_id: str,
        status: str,
        progress: Optional[int] = None,
        result: Optional[Dict] = None,
        error: Optional[str] = None,
        traceback: Optional[str] = None
    ):
        """
        Update submission status.
        
        Args:
            submission_id: Submission ID
            status: Status string
            progress: Progress percentage (0-100)
            result: Processing result
            error: Error message
            traceback: Error traceback
        """
        update_data = {
            'status': status,
            'updated_at': now()
        }
        
        if progress is not None:
            update_data['progress'] = progress
        
        if result is not None:
            update_data['result'] = result
        
        if error is not None:
            update_data['error'] = error
            update_data['error_traceback'] = traceback
        
        try:
            self.storage.update_submission(submission_id, update_data)
        except Exception as e:
            logger.error(f"Failed to update status for {submission_id}: {e}")


# Convenience function

async def process_submission_async(
    submission_id: str,
    storage_service: StorageService,
    orchestrator: Orchestrator
) -> Dict:
    """
    Process a single submission asynchronously.
    
    Convenience function for one-off processing without worker.
    
    Args:
        submission_id: Submission ID
        storage_service: Storage service
        orchestrator: Orchestrator instance
        
    Returns:
        dict: Processing result
    """
    from backend.services.queue_service import QueueService
    from backend.services.metrics_service import MetricsService
    
    queue = QueueService()
    metrics = MetricsService()
    
    worker = SubmissionWorker(
        storage_service,
        queue,
        metrics,
        orchestrator
    )
    
    return await worker.process_submission(submission_id)
