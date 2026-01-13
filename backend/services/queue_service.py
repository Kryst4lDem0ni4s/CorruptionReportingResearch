"""
Queue Service - Background job processing

Provides:
- In-memory job queue (no Redis/Celery)
- Async task processing
- Priority queue support
- Job status tracking
- Retry logic with exponential backoff
- Worker pool management
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

# Initialize logger
logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class JobPriority(int, Enum):
    """Job priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class Job:
    """
    Job data structure.
    """
    job_id: str
    task_name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[int] = None  # seconds
    
    def __lt__(self, other):
        """Compare jobs by priority (for priority queue)."""
        return self.priority.value > other.priority.value  # Higher priority first


class QueueService:
    """
    Queue Service - Background job processing.
    
    Features:
    - In-memory priority queue
    - Async job processing
    - Retry logic with exponential backoff
    - Job status tracking
    - Worker pool management
    - Job history
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        job_timeout: int = 300
    ):
        """
        Initialize queue service.
        
        Args:
            max_workers: Maximum concurrent workers
            max_queue_size: Maximum queue size
            job_timeout: Default job timeout in seconds
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.default_timeout = job_timeout
        
        # Job storage
        self.jobs: Dict[str, Job] = {}
        self.pending_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.running = False
        
        # Statistics
        self.stats = {
            'jobs_submitted': 0,
            'jobs_completed': 0,
            'jobs_failed': 0,
            'jobs_retried': 0,
            'total_processing_time': 0.0
        }
        
        logger.info(
            f"QueueService initialized "
            f"(workers={max_workers}, queue_size={max_queue_size})"
        )
    
    async def start(self) -> None:
        """Start worker pool."""
        if self.running:
            logger.warning("Queue service already running")
            return
        
        self.running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(worker_id=i))
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop worker pool gracefully."""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for workers to finish
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        
        self.workers.clear()
        logger.info("Queue service stopped")
    
    async def submit_job(
        self,
        task_name: str,
        func: Callable,
        *args,
        priority: JobPriority = JobPriority.NORMAL,
        max_retries: int = 3,
        timeout: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Submit job to queue.
        
        Args:
            task_name: Human-readable task name
            func: Function to execute
            *args: Positional arguments
            priority: Job priority
            max_retries: Maximum retry attempts
            timeout: Job timeout in seconds
            **kwargs: Keyword arguments
            
        Returns:
            str: Job ID
            
        Raises:
            ValueError: If queue is full
        """
        try:
            # Check queue size
            if self.pending_queue.qsize() >= self.max_queue_size:
                raise ValueError("Queue is full")
            
            # Create job
            job_id = str(uuid.uuid4())
            
            job = Job(
                job_id=job_id,
                task_name=task_name,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                max_retries=max_retries,
                timeout=timeout or self.default_timeout
            )
            
            # Store job
            self.jobs[job_id] = job
            
            # Add to queue
            await self.pending_queue.put(job)
            
            self.stats['jobs_submitted'] += 1
            
            logger.info(
                f"Job submitted: {job_id} ({task_name}) "
                f"[priority={priority.name}]"
            )
            
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to submit job: {e}")
            raise
    
    async def _worker(self, worker_id: int) -> None:
        """
        Worker coroutine that processes jobs from queue.
        
        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.running:
            try:
                # Get job from queue (with timeout)
                try:
                    job = await asyncio.wait_for(
                        self.pending_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process job
                await self._process_job(job, worker_id)
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_job(self, job: Job, worker_id: int) -> None:
        """
        Process a single job.
        
        Args:
            job: Job to process
            worker_id: Worker identifier
        """
        try:
            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            
            logger.info(
                f"Worker {worker_id} processing job {job.job_id} "
                f"({job.task_name})"
            )
            
            # Execute job with timeout
            try:
                if asyncio.iscoroutinefunction(job.func):
                    # Async function
                    result = await asyncio.wait_for(
                        job.func(*job.args, **job.kwargs),
                        timeout=job.timeout
                    )
                else:
                    # Sync function - run in executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(
                            None,
                            lambda: job.func(*job.args, **job.kwargs)
                        ),
                        timeout=job.timeout
                    )
                
                # Job completed successfully
                job.status = JobStatus.COMPLETED
                job.result = result
                job.completed_at = time.time()
                
                # Update stats
                processing_time = job.completed_at - job.started_at
                self.stats['jobs_completed'] += 1
                self.stats['total_processing_time'] += processing_time
                
                logger.info(
                    f"Job completed: {job.job_id} "
                    f"(time={processing_time:.2f}s)"
                )
                
            except asyncio.TimeoutError:
                raise TimeoutError(f"Job exceeded timeout of {job.timeout}s")
            
        except Exception as e:
            # Job failed
            error_msg = str(e)
            logger.error(f"Job failed: {job.job_id} - {error_msg}")
            
            job.error = error_msg
            
            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.RETRYING
                
                # Calculate backoff delay (exponential)
                backoff_delay = min(2 ** job.retry_count, 60)  # Max 60s
                
                logger.info(
                    f"Retrying job {job.job_id} "
                    f"(attempt {job.retry_count}/{job.max_retries}) "
                    f"in {backoff_delay}s"
                )
                
                # Schedule retry
                await asyncio.sleep(backoff_delay)
                await self.pending_queue.put(job)
                
                self.stats['jobs_retried'] += 1
            else:
                # Max retries exceeded
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                
                self.stats['jobs_failed'] += 1
                
                logger.error(
                    f"Job failed permanently: {job.job_id} "
                    f"(retries={job.retry_count})"
                )
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get job status.
        
        Args:
            job_id: Job identifier
            
        Returns:
            dict: Job status information or None
        """
        job = self.jobs.get(job_id)
        
        if not job:
            return None
        
        status_info = {
            'job_id': job.job_id,
            'task_name': job.task_name,
            'status': job.status.value,
            'priority': job.priority.name,
            'created_at': datetime.fromtimestamp(job.created_at).isoformat(),
            'retry_count': job.retry_count,
            'max_retries': job.max_retries
        }
        
        if job.started_at:
            status_info['started_at'] = datetime.fromtimestamp(job.started_at).isoformat()
        
        if job.completed_at:
            status_info['completed_at'] = datetime.fromtimestamp(job.completed_at).isoformat()
            status_info['duration'] = job.completed_at - job.started_at
        
        if job.status == JobStatus.COMPLETED:
            status_info['result'] = job.result
        elif job.status == JobStatus.FAILED:
            status_info['error'] = job.error
        
        return status_info
    
    def get_job_result(self, job_id: str) -> Optional[Any]:
        """
        Get job result (if completed).
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job result or None
        """
        job = self.jobs.get(job_id)
        
        if job and job.status == JobStatus.COMPLETED:
            return job.result
        
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel pending job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            bool: True if cancelled, False if not found or already running
        """
        job = self.jobs.get(job_id)
        
        if not job:
            return False
        
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            logger.info(f"Job cancelled: {job_id}")
            return True
        
        return False
    
    def get_queue_stats(self) -> Dict:
        """
        Get queue statistics.
        
        Returns:
            dict: Queue statistics
        """
        # Count jobs by status
        status_counts = defaultdict(int)
        for job in self.jobs.values():
            status_counts[job.status.value] += 1
        
        # Calculate average processing time
        avg_processing_time = 0.0
        if self.stats['jobs_completed'] > 0:
            avg_processing_time = (
                self.stats['total_processing_time'] / 
                self.stats['jobs_completed']
            )
        
        stats = {
            'running': self.running,
            'workers': len(self.workers),
            'queue_size': self.pending_queue.qsize(),
            'max_queue_size': self.max_queue_size,
            'total_jobs': len(self.jobs),
            'jobs_by_status': dict(status_counts),
            'jobs_submitted': self.stats['jobs_submitted'],
            'jobs_completed': self.stats['jobs_completed'],
            'jobs_failed': self.stats['jobs_failed'],
            'jobs_retried': self.stats['jobs_retried'],
            'avg_processing_time': round(avg_processing_time, 2),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return stats
    
    def cleanup_old_jobs(self, max_age_seconds: int = 86400) -> int:
        """
        Remove old completed/failed jobs.
        
        Args:
            max_age_seconds: Maximum age in seconds (default: 24 hours)
            
        Returns:
            int: Number of jobs removed
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        jobs_to_remove = []
        
        for job_id, job in self.jobs.items():
            # Remove if completed/failed and old
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                if job.completed_at and job.completed_at < cutoff_time:
                    jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del self.jobs[job_id]
        
        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")
        
        return len(jobs_to_remove)
