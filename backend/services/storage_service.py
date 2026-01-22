"""
Storage Service - JSON-based submission storage

Provides:
- Atomic file operations with locking
- Submission CRUD operations
- Validator pool management
- Index management for fast lookup
- Directory sharding for scalability
- Archive functionality
"""

import json
import logging
import threading
import os
import time

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from filelock import FileLock

# Initialize logger
logger = logging.getLogger(__name__)


class StorageService:
    """
    Storage Service for JSON-based data persistence.
    
    Features:
    - Thread-safe atomic writes with file locking
    - Submission storage with automatic indexing
    - Validator pool persistence
    - Archive functionality
    - Fast lookup via index file
    - Directory sharding by year/month
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize storage service.
        
        Args:
            data_dir: Base data directory (defaults to backend/data)
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent / "data"
        
        # Create directory structure
        self.submissions_dir = self.data_dir / "submissions"
        self.evidence_dir = self.data_dir / "evidence"
        self.reports_dir = self.data_dir / "reports"
        self.cache_dir = self.data_dir / "cache"
        self.archive_dir = self.data_dir / "archive"
        
        # Initialize directories
        self._initialize_directories()
        
        # Index file
        self.index_file = self.data_dir / "index.json"
        self.index_lock = threading.RLock()
        
        # Validators file
        self.validators_file = self.data_dir / "validators.json"
        
        logger.info(f"StorageService initialized (data_dir={self.data_dir})")
    
    def _initialize_directories(self) -> None:
        """Create all required directories."""
        directories = [
            self.data_dir,
            self.submissions_dir,
            self.evidence_dir,
            self.reports_dir,
            self.cache_dir,
            self.archive_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Storage directories initialized")
    
    def save_submission(self, submission_id: str, data: Dict) -> None:
        """
        Save submission data to JSON file with atomic write.
        
        Args:
            submission_id: Unique submission identifier
            data: Submission data dictionary
            
        Raises:
            IOError: If save fails
        """
        try:
            # Ensure submission_id is in data
            data['id'] = submission_id
            
            # Add/update timestamp
            if 'timestamp_updated' not in data:
                data['timestamp_updated'] = datetime.utcnow().isoformat()
            
            # File path
            file_path = self.submissions_dir / f"{submission_id}.json"
            lock_path = self.submissions_dir / f"{submission_id}.json.lock"
            
            # Atomic write with file lock
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                # Write to temporary file first
                temp_path = file_path.with_suffix('.tmp')
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename with Windows support
                self._safe_replace(temp_path, file_path)
            
            # Update index
            self._update_index(submission_id, data)
            
            logger.debug(f"Saved submission {submission_id}")
            
        except Exception as e:
            logger.error(f"Failed to save submission {submission_id}: {e}")
            raise IOError(f"Failed to save submission: {str(e)}")
    
    def load_submission(self, submission_id: str) -> Optional[Dict]:
        """
        Load submission data from JSON file.
        
        Args:
            submission_id: Unique submission identifier
            
        Returns:
            dict: Submission data, or None if not found
        """
        try:
            file_path = self.submissions_dir / f"{submission_id}.json"
            
            if not file_path.exists():
                logger.debug(f"Submission {submission_id} not found")
                return None
            
            lock_path = self.submissions_dir / f"{submission_id}.json.lock"
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to load submission {submission_id}: {e}")
            return None

    def get_submission(self, submission_id: str) -> Optional[Dict]:
        """Alias for load_submission."""
        return self.load_submission(submission_id)

    def update_submission(self, submission_id: str, updates: Dict) -> None:
        """
        Update submission with partial data.
        
        Args:
            submission_id: Submission identifier
            updates: Dictionary of updates to apply
        """
        try:
            lock_path = self.submissions_dir / f"{submission_id}.json.lock"
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                submission = self.load_submission(submission_id)
                if submission:
                    submission.update(updates)
                    self.save_submission(submission_id, submission)
        except Exception as e:
            logger.error(f"Failed to update submission {submission_id}: {e}")
    
    def delete_submission(self, submission_id: str) -> bool:
        """
        Delete submission from storage.
        
        Args:
            submission_id: Unique submission identifier
            
        Returns:
            bool: True if deleted, False otherwise
        """
        try:
            file_path = self.submissions_dir / f"{submission_id}.json"
            
            if not file_path.exists():
                return False
            
            lock_path = self.submissions_dir / f"{submission_id}.json.lock"
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                file_path.unlink()
            
            # Remove lock file
            if lock_path.exists():
                lock_path.unlink()
            
            # Remove from index
            self._remove_from_index(submission_id)
            
            logger.info(f"Deleted submission {submission_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete submission {submission_id}: {e}")
            return False
    
    def get_all_submissions(self) -> List[Dict]:
        """
        Get all submissions from storage.
        
        Returns:
            list: List of all submission dictionaries
        """
        submissions = []
        
        try:
            # Use index for fast lookup
            index = self._load_index()
            
            for submission_id in index.keys():
                submission = self.load_submission(submission_id)
                if submission:
                    submissions.append(submission)
            
            logger.debug(f"Loaded {len(submissions)} submissions")
            
        except Exception as e:
            logger.error(f"Failed to get all submissions: {e}")
        
        return submissions
    
    def update_submission_status(self, submission_id: str, status: str) -> None:
        """
        Update submission status.
        
        Args:
            submission_id: Submission identifier
            status: New status value
        """
        submission = self.load_submission(submission_id)
        
        if submission:
            submission['status'] = status
            submission['status_updated'] = datetime.utcnow().isoformat()
            self.save_submission(submission_id, submission)
    
    def archive_submission(self, submission_id: str) -> bool:
        """
        Archive old submission by moving to archive directory.
        
        Args:
            submission_id: Submission identifier
            
        Returns:
            bool: True if archived, False otherwise
        """
        try:
            source = self.submissions_dir / f"{submission_id}.json"
            
            if not source.exists():
                return False
            
            # Create archive directory if needed
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            
            # Move to archive
            dest = self.archive_dir / f"{submission_id}.json"
            
            lock_path = self.submissions_dir / f"{submission_id}.json.lock"
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                source.rename(dest)
            
            # Clean up lock file
            if lock_path.exists():
                lock_path.unlink()
            
            # Remove from index
            self._remove_from_index(submission_id)
            
            logger.info(f"Archived submission {submission_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to archive submission {submission_id}: {e}")
            return False
    
    def save_validators(self, validators: List[Dict]) -> None:
        """
        Save validator pool to storage.
        
        Args:
            validators: List of validator dictionaries
        """
        try:
            lock_path = self.validators_file.with_suffix('.lock')
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                temp_path = self.validators_file.with_suffix('.tmp')
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(validators, f, indent=2, ensure_ascii=False)
                
                temp_path.replace(self.validators_file)
            
            logger.debug(f"Saved {len(validators)} validators")
            
        except Exception as e:
            logger.error(f"Failed to save validators: {e}")
            raise IOError(f"Failed to save validators: {str(e)}")
    
    def load_validators(self) -> List[Dict]:
        """
        Load validator pool from storage.
        
        Returns:
            list: List of validator dictionaries
        """
        try:
            if not self.validators_file.exists():
                logger.debug("Validators file not found, returning empty list")
                return []
            
            lock_path = self.validators_file.with_suffix('.lock')
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                with open(self.validators_file, 'r', encoding='utf-8') as f:
                    validators = json.load(f)
            
            logger.debug(f"Loaded {len(validators)} validators")
            return validators
            
        except Exception as e:
            logger.error(f"Failed to load validators: {e}")
            return []
    
    def health_check(self) -> bool:
        """
        Check if storage is accessible and healthy.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        try:
            # Check if directories exist
            directories_ok = (
                self.data_dir.exists() and
                self.submissions_dir.exists() and
                self.evidence_dir.exists()
            )
            
            if not directories_ok:
                logger.warning("Storage directories not accessible")
                return False
            
            # Test write/read
            test_file = self.data_dir / ".health_check"
            test_data = {"timestamp": datetime.utcnow().isoformat()}
            
            with open(test_file, 'w') as f:
                json.dump(test_data, f)
            
            with open(test_file, 'r') as f:
                json.load(f)
            
            test_file.unlink()
            
            logger.debug("Storage health check passed")
            return True
            
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
            return False
    
    def _update_index(self, submission_id: str, data: Dict) -> None:
        """
        Update index file with submission metadata.
        
        Args:
            submission_id: Submission identifier
            data: Submission data
        """
        try:
            with self.index_lock:
                index = self._load_index()
                
                # Add/update entry
                index[submission_id] = {
                    'pseudonym': data.get('pseudonym'),
                    'evidence_type': data.get('evidence_type'),
                    'status': data.get('status'),
                    'timestamp_submission': data.get('timestamp_submission'),
                    'timestamp_updated': data.get('timestamp_updated')
                }
                
                # Save index
                self._save_index(index)
                
        except Exception as e:
            logger.warning(f"Failed to update index: {e}")
    
    def _remove_from_index(self, submission_id: str) -> None:
        """
        Remove submission from index.
        
        Args:
            submission_id: Submission identifier
        """
        try:
            with self.index_lock:
                index = self._load_index()
                
                if submission_id in index:
                    del index[submission_id]
                    self._save_index(index)
                    
        except Exception as e:
            logger.warning(f"Failed to remove from index: {e}")
    
    def _load_index(self) -> Dict:
        """
        Load index file.
        
        Returns:
            dict: Index data
        """
        try:
            with self.index_lock:
                if not self.index_file.exists():
                    return {}
                
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
                
        except Exception as e:
            logger.warning(f"Failed to load index: {e}")
            return {}
    
    def _save_index(self, index: Dict) -> None:
        """
        Save index file.
        
        Args:
            index: Index data
        """
        try:
            temp_path = self.index_file.with_suffix('.tmp')
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(index, f, indent=2, ensure_ascii=False)
            
            self._safe_replace(temp_path, self.index_file)
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    def get_storage_statistics(self) -> Dict:
        """
        Get storage statistics.
        
        Returns:
            dict: Storage statistics
        """
        try:
            index = self._load_index()
            
            # Count by status
            status_counts = {}
            for entry in index.values():
                status = entry.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # Count by evidence type
            type_counts = {}
            for entry in index.values():
                evidence_type = entry.get('evidence_type', 'unknown')
                type_counts[evidence_type] = type_counts.get(evidence_type, 0) + 1
            
            # Calculate storage size
            total_size = 0
            for file_path in self.submissions_dir.glob("*.json"):
                total_size += file_path.stat().st_size
            
            stats = {
                'total_submissions': len(index),
                'status_distribution': status_counts,
                'type_distribution': type_counts,
                'storage_size_bytes': total_size,
                'storage_size_mb': round(total_size / (1024 * 1024), 2),
                'data_directory': str(self.data_dir),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage statistics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def get_evidence_path(
        self,
        submission_id: str,
        filename: str,
        create_dirs: bool = True
    ) -> Path:
        """
        Get path for evidence file with year/month sharding.
        
        Args:
            submission_id: Submission identifier
            filename: Evidence filename
            create_dirs: Create directories if they don't exist
            
        Returns:
            Path: Full path for evidence file
        """
        now = datetime.utcnow()
        year = now.strftime('%Y')
        month = now.strftime('%m')
        
        # Sharded directory structure
        evidence_path = self.evidence_dir / year / month
        
        if create_dirs:
            evidence_path.mkdir(parents=True, exist_ok=True)
        
        return evidence_path / filename
    
    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """
        Cleanup old cache files.
        
        Args:
            max_age_hours: Maximum age of cache files in hours
            
        Returns:
            int: Number of files deleted
        """
        try:
            from datetime import timedelta
            
            cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
            deleted = 0
            
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    
                    if file_time < cutoff_time:
                        file_path.unlink()
                        deleted += 1
            
            logger.info(f"Cleaned up {deleted} cache files")
            return deleted
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
            return 0
    
    def save_evidence_file(
        self,
        submission_id: str,
        filename: str,
        content: bytes,
        evidence_type: str
    ) -> Path:
        """
        Save evidence file content to storage.
        
        Args:
            submission_id: Submission identifier
            filename: Original filename
            content: File content bytes
            evidence_type: Type of evidence
            
        Returns:
            Path: Path to saved file
        """
        try:
            # Get evidence path with sharding
            safe_filename = self._sanitize_filename(filename)
            file_path = self.get_evidence_path(submission_id, safe_filename)
            
            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file content
            with open(file_path, 'wb') as f:
                f.write(content)
            
            logger.debug(f"Saved evidence file: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save evidence file: {e}")
            raise IOError(f"Failed to save evidence file: {str(e)}")
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        import re
        # Remove path components
        filename = Path(filename).name
        # Replace unsafe characters
        safe_name = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        # Limit length
        if len(safe_name) > 255:
            name, ext = safe_name.rsplit('.', 1) if '.' in safe_name else (safe_name, '')
            safe_name = name[:250] + ('.' + ext if ext else '')
        return safe_name or 'unnamed_file'
    
    def _safe_replace(self, src: Path, dst: Path, retries: int = 3) -> None:
        """
        Safely replace file, handling Windows locking issues.
        
        Args:
            src: Source path
            dst: Destination path
            retries: Number of retries on PermissionError
        """
        for i in range(retries):
            try:
                src.replace(dst)
                return
            except PermissionError:
                if i == retries - 1:
                    raise
                
                # On Windows, destination might be open or locked temporarily
                # Try to unlink if it exists
                try:
                    if dst.exists():
                        dst.unlink()
                except Exception:
                    pass  # Ignore unlink errors, retry replace
                
                time.sleep(0.1)
