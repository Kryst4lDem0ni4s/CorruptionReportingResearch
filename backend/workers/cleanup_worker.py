"""
Cleanup Worker - Data maintenance and cleanup

Handles:
- Old submission removal
- Temporary file cleanup
- Cache purging
- Log rotation
"""

import logging
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from backend.services.storage_service import StorageService
from backend.services.metrics_service import MetricsService
from backend.utils import get_logger, now, TimeUtils

# Initialize logger
logger = get_logger(__name__)


class CleanupWorker:
    """
    Background worker for data cleanup and maintenance.
    
    Features:
    - Remove old submissions
    - Clean temporary files
    - Purge cache
    - Archive old reports
    - Maintain storage limits
    """
    
    def __init__(
        self,
        storage_service: StorageService,
        metrics_service: MetricsService,
        data_dir: Path,
        retention_days: int = 90,
        cache_max_age_hours: int = 24,
        max_storage_gb: float = 10.0
    ):
        """
        Initialize cleanup worker.
        
        Args:
            storage_service: Storage service instance
            metrics_service: Metrics service instance
            data_dir: Data directory path
            retention_days: Days to retain submissions
            cache_max_age_hours: Max age for cache files (hours)
            max_storage_gb: Maximum storage in GB
        """
        self.storage = storage_service
        # self.metrics = metrics_service
        import types
        # If caller passed the module (or the class), instantiate the service.
        if isinstance(metrics_service, types.ModuleType):
            if hasattr(metrics_service, 'MetricsService'):
                metrics_service = metrics_service.MetricsService()
            else:
                raise TypeError("Provided metrics_service is a module without 'MetricsService' class")
        elif isinstance(metrics_service, type):
            # A class was passed (MetricsService), instantiate it
            metrics_service = metrics_service()
        # Final check: must provide an instance with record_gauge
        if not hasattr(metrics_service, 'record_gauge'):
            raise TypeError("metrics_service must be an instance exposing 'record_gauge'")
        self.metrics = metrics_service
        self.data_dir = Path(data_dir)
        self.retention_days = retention_days
        self.cache_max_age_hours = cache_max_age_hours
        self.max_storage_gb = max_storage_gb
        
        self.is_running = False
        
        logger.info(
            f"CleanupWorker initialized - "
            f"retention={retention_days}d, cache_max={cache_max_age_hours}h"
        )
    
    def start(self, interval_hours: float = 24.0):
        """
        Start the cleanup worker.
        
        Runs cleanup tasks periodically.
        
        Args:
            interval_hours: Hours between cleanup runs
        """
        if self.is_running:
            logger.warning("Cleanup worker already running")
            return
        
        self.is_running = True
        logger.info(f"CleanupWorker started (interval={interval_hours}h)")
        
        try:
            while self.is_running:
                # Run cleanup
                self.run_cleanup()
                
                # Sleep until next run
                if self.is_running:
                    time.sleep(interval_hours * 3600)
        
        except KeyboardInterrupt:
            logger.info("Cleanup worker interrupted")
        except Exception as e:
            logger.error(f"Cleanup worker error: {e}", exc_info=True)
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop the cleanup worker."""
        logger.info("Stopping CleanupWorker...")
        self.is_running = False
    
    def run_cleanup(self) -> Dict:
        """
        Run all cleanup tasks.
        
        Returns:
            dict: Cleanup statistics
        """
        logger.info("Starting cleanup tasks...")
        start_time = now()
        
        stats = {
            'start_time': start_time,
            'tasks': {}
        }
        
        try:
            # Task 1: Clean old submissions
            logger.info("Cleaning old submissions...")
            submissions_stats = self._clean_old_submissions()
            stats['tasks']['old_submissions'] = submissions_stats
            
            # Task 2: Clean cache
            logger.info("Cleaning cache...")
            cache_stats = self._clean_cache()
            stats['tasks']['cache'] = cache_stats
            
            # Task 3: Clean temporary files
            logger.info("Cleaning temporary files...")
            temp_stats = self._clean_temp_files()
            stats['tasks']['temp_files'] = temp_stats
            
            # Task 4: Archive old reports
            logger.info("Archiving old reports...")
            archive_stats = self._archive_old_reports()
            stats['tasks']['archive'] = archive_stats
            
            # Task 5: Check storage limits
            logger.info("Checking storage limits...")
            storage_stats = self._check_storage_limits()
            stats['tasks']['storage'] = storage_stats
            
            # Calculate totals
            stats['total_files_removed'] = sum(
                task.get('files_removed', 0)
                for task in stats['tasks'].values()
            )
            stats['total_space_freed_mb'] = sum(
                task.get('space_freed_mb', 0)
                for task in stats['tasks'].values()
            )
            
            duration = now() - start_time
            stats['duration_seconds'] = duration
            
            logger.info(
                f"Cleanup completed in {duration:.2f}s - "
                f"Files removed: {stats['total_files_removed']}, "
                f"Space freed: {stats['total_space_freed_mb']:.2f} MB"
            )
            
            # Record metrics
            self.metrics.record_gauge('cleanup_files_removed', stats['total_files_removed'])
            self.metrics.record_gauge('cleanup_space_freed_mb', stats['total_space_freed_mb'])
            
            return stats
        
        except Exception as e:
            logger.error(f"Cleanup failed: {e}", exc_info=True)
            stats['error'] = str(e)
            return stats
    
    def _clean_old_submissions(self) -> Dict:
        """
        Remove submissions older than retention period.
        
        Returns:
            dict: Cleanup statistics
        """
        cutoff_time = now() - (self.retention_days * 86400)
        
        removed_count = 0
        space_freed = 0
        errors = []
        
        try:
            submissions_dir = self.data_dir / 'submissions'
            
            if not submissions_dir.exists():
                return {'files_removed': 0, 'space_freed_mb': 0}
            
            # Iterate through submission files
            for submission_file in submissions_dir.glob('*.json'):
                try:
                    # Check file modification time
                    file_mtime = submission_file.stat().st_mtime
                    
                    if file_mtime < cutoff_time:
                        # Get submission ID
                        submission_id = submission_file.stem
                        
                        # Get file size
                        file_size = submission_file.stat().st_size
                        
                        # Remove associated evidence files
                        submission = self.storage.get_submission(submission_id)
                        if submission:
                            evidence_paths = submission.get('evidence_paths', [])
                            for path in evidence_paths:
                                evidence_file = Path(path)
                                if evidence_file.exists():
                                    file_size += evidence_file.stat().st_size
                                    evidence_file.unlink()
                        
                        # Remove submission file
                        submission_file.unlink()
                        
                        removed_count += 1
                        space_freed += file_size
                        
                        logger.debug(f"Removed old submission: {submission_id}")
                
                except Exception as e:
                    logger.error(f"Error removing {submission_file}: {e}")
                    errors.append(str(e))
            
            return {
                'files_removed': removed_count,
                'space_freed_mb': space_freed / (1024 * 1024),
                'errors': errors
            }
        
        except Exception as e:
            logger.error(f"Error in _clean_old_submissions: {e}")
            return {'files_removed': 0, 'space_freed_mb': 0, 'error': str(e)}
    
    def _clean_cache(self) -> Dict:
        """
        Clean old cache files.
        
        Returns:
            dict: Cleanup statistics
        """
        cutoff_time = now() - (self.cache_max_age_hours * 3600)
        
        removed_count = 0
        space_freed = 0
        
        try:
            cache_dir = self.data_dir / 'cache'
            
            if not cache_dir.exists():
                return {'files_removed': 0, 'space_freed_mb': 0}
            
            # Remove old cache files
            for cache_file in cache_dir.rglob('*'):
                if cache_file.is_file():
                    file_mtime = cache_file.stat().st_mtime
                    
                    if file_mtime < cutoff_time:
                        file_size = cache_file.stat().st_size
                        cache_file.unlink()
                        
                        removed_count += 1
                        space_freed += file_size
            
            return {
                'files_removed': removed_count,
                'space_freed_mb': space_freed / (1024 * 1024)
            }
        
        except Exception as e:
            logger.error(f"Error in _clean_cache: {e}")
            return {'files_removed': 0, 'space_freed_mb': 0, 'error': str(e)}
    
    def _clean_temp_files(self) -> Dict:
        """
        Clean temporary files.
        
        Returns:
            dict: Cleanup statistics
        """
        removed_count = 0
        space_freed = 0
        
        try:
            # Clean Python __pycache__ directories
            for pycache_dir in self.data_dir.parent.rglob('__pycache__'):
                if pycache_dir.is_dir():
                    dir_size = sum(
                        f.stat().st_size
                        for f in pycache_dir.rglob('*')
                        if f.is_file()
                    )
                    
                    shutil.rmtree(pycache_dir)
                    space_freed += dir_size
                    removed_count += 1
            
            # Clean .pyc files
            backend_dir = self.data_dir.parent / 'backend'
            if backend_dir.exists():
                for pyc_file in backend_dir.rglob('*.pyc'):
                    file_size = pyc_file.stat().st_size
                    pyc_file.unlink()
                    space_freed += file_size
                    removed_count += 1
            
            return {
                'files_removed': removed_count,
                'space_freed_mb': space_freed / (1024 * 1024)
            }
        
        except Exception as e:
            logger.error(f"Error in _clean_temp_files: {e}")
            return {'files_removed': 0, 'space_freed_mb': 0, 'error': str(e)}
    
    def _archive_old_reports(self) -> Dict:
        """
        Archive old report files.
        
        Moves reports older than 30 days to archive.
        
        Returns:
            dict: Archive statistics
        """
        cutoff_time = now() - (30 * 86400)  # 30 days
        
        archived_count = 0
        
        try:
            reports_dir = self.data_dir / 'reports'
            archive_dir = self.data_dir / 'reports_archive'
            
            if not reports_dir.exists():
                return {'files_archived': 0}
            
            # Create archive directory
            archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Archive old reports
            for report_file in reports_dir.glob('*.pdf'):
                file_mtime = report_file.stat().st_mtime
                
                if file_mtime < cutoff_time:
                    # Move to archive
                    archive_path = archive_dir / report_file.name
                    report_file.rename(archive_path)
                    
                    archived_count += 1
            
            return {'files_archived': archived_count}
        
        except Exception as e:
            logger.error(f"Error in _archive_old_reports: {e}")
            return {'files_archived': 0, 'error': str(e)}
    
    def _check_storage_limits(self) -> Dict:
        """
        Check storage usage and warn if limit exceeded.
        
        Returns:
            dict: Storage statistics
        """
        try:
            # Calculate total storage usage
            total_size = sum(
                f.stat().st_size
                for f in self.data_dir.rglob('*')
                if f.is_file()
            )
            
            total_size_gb = total_size / (1024 ** 3)
            
            usage_percent = (total_size_gb / self.max_storage_gb) * 100
            
            logger.info(
                f"Storage usage: {total_size_gb:.2f} GB / "
                f"{self.max_storage_gb:.2f} GB ({usage_percent:.1f}%)"
            )
            
            if total_size_gb > self.max_storage_gb:
                logger.warning(
                    f"Storage limit exceeded! "
                    f"Using {total_size_gb:.2f} GB / {self.max_storage_gb:.2f} GB"
                )
            
            # Record metric
            self.metrics.record_gauge('storage_usage_gb', total_size_gb)
            self.metrics.record_gauge('storage_usage_percent', usage_percent)
            
            return {
                'total_size_gb': total_size_gb,
                'limit_gb': self.max_storage_gb,
                'usage_percent': usage_percent,
                'limit_exceeded': total_size_gb > self.max_storage_gb
            }
        
        except Exception as e:
            logger.error(f"Error in _check_storage_limits: {e}")
            return {'error': str(e)}


# Convenience function

def run_cleanup(
    data_dir: Path,
    retention_days: int = 90,
    cache_max_age_hours: int = 24
) -> Dict:
    """
    Run cleanup tasks once.
    
    Args:
        data_dir: Data directory path
        retention_days: Days to retain submissions
        cache_max_age_hours: Max age for cache files
        
    Returns:
        dict: Cleanup statistics
    """
    from backend.services.storage_service import StorageService
    from backend.services.metrics_service import MetricsService
    
    storage = StorageService(data_dir=data_dir)
    metrics = MetricsService()
    
    worker = CleanupWorker(
        storage,
        metrics,
        data_dir,
        retention_days,
        cache_max_age_hours
    )
    
    return worker.run_cleanup()
