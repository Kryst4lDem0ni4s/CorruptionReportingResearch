"""
File Utils - Atomic file operations and locking

Provides:
- Atomic file writes (write to temp, then rename)
- File locking for concurrent access
- Safe directory operations
- File size utilities
- Path sanitization
"""

import portalocker
import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Union

# Initialize logger
logger = logging.getLogger(__name__)


class FileUtils:
    """
    File utilities for safe file operations.
    
    Features:
    - Atomic writes (prevents partial writes)
    - File locking (prevents race conditions)
    - Safe directory operations
    - JSON file handling
    - Path sanitization
    """
    
    @staticmethod
    def atomic_write(
        file_path: Path,
        data: Union[str, bytes],
        mode: str = 'w',
        encoding: str = 'utf-8'
    ) -> None:
        """
        Write file atomically using temp file + rename.
        
        This prevents partial writes and ensures data integrity.
        The file is written to a temporary location first, then
        atomically renamed to the target path.
        
        Args:
            file_path: Target file path
            data: Data to write (str or bytes)
            mode: Write mode ('w' for text, 'wb' for binary)
            encoding: Text encoding (for text mode)
            
        Raises:
            IOError: If write fails
        """
        try:
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temp file in same directory (for atomic rename)
            temp_fd, temp_path = tempfile.mkstemp(
                dir=file_path.parent,
                prefix=f'.tmp_{file_path.name}_',
                suffix='.tmp'
            )
            
            try:
                # Write to temp file
                if 'b' in mode:
                    # Binary mode
                    os.write(temp_fd, data)
                else:
                    # Text mode
                    if isinstance(data, bytes):
                        data = data.decode(encoding)
                    os.write(temp_fd, data.encode(encoding))
                
                # Ensure data is written to disk
                os.fsync(temp_fd)
                
                # Close temp file
                os.close(temp_fd)
                
                # Atomic rename
                # On POSIX systems, this is atomic
                # On Windows, need to handle existing file
                if os.name == 'nt' and file_path.exists():
                    # Windows: remove existing file first
                    file_path.unlink()
                
                Path(temp_path).rename(file_path)
                
                logger.debug(f"Atomic write completed: {file_path}")
                
            except Exception as e:
                # Cleanup temp file on error
                try:
                    os.close(temp_fd)
                except:
                    pass
                
                if Path(temp_path).exists():
                    Path(temp_path).unlink()
                
                raise IOError(f"Atomic write failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"Atomic write failed for {file_path}: {e}")
            raise IOError(f"Failed to write file: {str(e)}")
    
    @staticmethod
    def atomic_write_json(
        file_path: Path,
        data: Any,
        indent: Optional[int] = 2,
        ensure_ascii: bool = False
    ) -> None:
        """
        Write JSON file atomically.
        
        Args:
            file_path: Target file path
            data: Data to serialize as JSON
            indent: JSON indentation (None for compact)
            ensure_ascii: Escape non-ASCII characters
            
        Raises:
            IOError: If write fails
            ValueError: If data cannot be serialized
        """
        try:
            # Serialize to JSON
            json_data = json.dumps(
                data,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=False
            )
            
            # Write atomically
            FileUtils.atomic_write(file_path, json_data, mode='w')
            
            logger.debug(f"Atomic JSON write completed: {file_path}")
            
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization failed: {e}")
            raise ValueError(f"Cannot serialize data to JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Atomic JSON write failed: {e}")
            raise IOError(f"Failed to write JSON file: {str(e)}")
    
    @staticmethod
    @contextmanager
    def file_lock(
        file_path: Path,
        timeout: Optional[float] = 10.0,
        shared: bool = False
    ) -> Generator[None, None, None]:
        """
        Context manager for file locking.
        
        Uses portalocker (POSIX) or msvcrt (Windows) for cross-process locking.
        
        Args:
            file_path: File to lock
            timeout: Lock timeout in seconds (None = wait forever)
            shared: Shared lock (read) vs exclusive lock (write)
            
        Yields:
            None
            
        Raises:
            TimeoutError: If lock cannot be acquired within timeout
            IOError: If locking fails
            
        Example:
            with FileUtils.file_lock(Path('data.json')):
                # File is locked here
                data = json.load(open('data.json'))
                # File is automatically unlocked on exit
        """
        # Ensure file exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if not file_path.exists():
            file_path.touch()
        
        # Open file
        lock_file = open(file_path, 'r+')
        
        try:
            # Determine lock type
            if shared:
                lock_type = portalocker.LOCK_SH  # Shared lock (read)
            else:
                lock_type = portalocker.LOCK_EX  # Exclusive lock (write)
            
            # Try to acquire lock
            start_time = time.time()
            
            while True:
                try:
                    # Non-blocking lock attempt
                    portalocker.flock(lock_file.fileno(), lock_type | portalocker.LOCK_NB)
                    logger.debug(f"Lock acquired: {file_path}")
                    break
                    
                except IOError as e:
                    # Lock is held by another process
                    if timeout is not None:
                        elapsed = time.time() - start_time
                        if elapsed >= timeout:
                            raise TimeoutError(
                                f"Could not acquire lock on {file_path} "
                                f"within {timeout}s"
                            )
                    
                    # Wait a bit and retry
                    time.sleep(0.01)
            
            # Yield control to caller
            yield
            
        finally:
            # Release lock
            try:
                portalocker.flock(lock_file.fileno(), portalocker.LOCK_UN)
                logger.debug(f"Lock released: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to release lock: {e}")
            
            # Close file
            lock_file.close()
    
    @staticmethod
    def safe_read_json(
        file_path: Path,
        default: Optional[Any] = None
    ) -> Any:
        """
        Safely read JSON file with locking.
        
        Args:
            file_path: File to read
            default: Default value if file doesn't exist or is invalid
            
        Returns:
            Parsed JSON data or default value
        """
        try:
            if not file_path.exists():
                logger.debug(f"File not found, using default: {file_path}")
                return default
            
            # Read with shared lock
            with FileUtils.file_lock(file_path, shared=True):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            
            logger.debug(f"JSON read completed: {file_path}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in {file_path}: {e}")
            return default
        except Exception as e:
            logger.error(f"Failed to read JSON from {file_path}: {e}")
            return default
    
    @staticmethod
    def safe_write_json(
        file_path: Path,
        data: Any,
        indent: Optional[int] = 2
    ) -> bool:
        """
        Safely write JSON file with locking and atomic write.
        
        Args:
            file_path: File to write
            data: Data to serialize
            indent: JSON indentation
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create lock file path
            lock_path = file_path.parent / f'.{file_path.name}.lock'
            
            # Write with exclusive lock
            with FileUtils.file_lock(lock_path, shared=False):
                FileUtils.atomic_write_json(file_path, data, indent=indent)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write JSON to {file_path}: {e}")
            return False
    
    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """
        Ensure directory exists, create if necessary.
        
        Args:
            directory: Directory path
        """
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
            
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise IOError(f"Cannot create directory: {str(e)}")
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: File path
            
        Returns:
            int: File size in bytes
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return file_path.stat().st_size
    
    @staticmethod
    def get_human_readable_size(size_bytes: int) -> str:
        """
        Convert bytes to human-readable format.
        
        Args:
            size_bytes: Size in bytes
            
        Returns:
            str: Human-readable size (e.g., "1.5 MB")
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.1f} PB"
    
    @staticmethod
    def copy_file_safe(source: Path, destination: Path) -> None:
        """
        Copy file safely with atomic write.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Raises:
            FileNotFoundError: If source doesn't exist
            IOError: If copy fails
        """
        try:
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            
            # Read source
            with open(source, 'rb') as f:
                data = f.read()
            
            # Write to destination atomically
            FileUtils.atomic_write(destination, data, mode='wb')
            
            logger.debug(f"File copied: {source} → {destination}")
            
        except Exception as e:
            logger.error(f"Failed to copy {source} to {destination}: {e}")
            raise IOError(f"File copy failed: {str(e)}")
    
    @staticmethod
    def move_file_safe(source: Path, destination: Path) -> None:
        """
        Move file safely.
        
        Args:
            source: Source file path
            destination: Destination file path
            
        Raises:
            FileNotFoundError: If source doesn't exist
            IOError: If move fails
        """
        try:
            if not source.exists():
                raise FileNotFoundError(f"Source file not found: {source}")
            
            # Ensure destination directory exists
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # Try atomic rename first (fast if same filesystem)
            try:
                source.rename(destination)
                logger.debug(f"File moved (renamed): {source} → {destination}")
                return
            except OSError:
                # Cross-filesystem move, need to copy + delete
                pass
            
            # Copy then delete
            FileUtils.copy_file_safe(source, destination)
            source.unlink()
            
            logger.debug(f"File moved (copied): {source} → {destination}")
            
        except Exception as e:
            logger.error(f"Failed to move {source} to {destination}: {e}")
            raise IOError(f"File move failed: {str(e)}")
    
    @staticmethod
    def delete_file_safe(file_path: Path) -> bool:
        """
        Delete file safely.
        
        Args:
            file_path: File to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        try:
            if not file_path.exists():
                return False
            
            file_path.unlink()
            logger.debug(f"File deleted: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
            return False
    
    @staticmethod
    def sanitize_path(path: str, base_dir: Optional[Path] = None) -> Path:
        """
        Sanitize file path to prevent directory traversal.
        
        Args:
            path: Path to sanitize
            base_dir: Base directory to restrict to
            
        Returns:
            Path: Sanitized path
            
        Raises:
            ValueError: If path tries to escape base directory
        """
        # Convert to Path and resolve
        sanitized = Path(path).resolve()
        
        # If base_dir specified, ensure path is within it
        if base_dir:
            base_dir = base_dir.resolve()
            
            # Check if path is within base_dir
            try:
                sanitized.relative_to(base_dir)
            except ValueError:
                raise ValueError(
                    f"Path {path} is outside base directory {base_dir}"
                )
        
        return sanitized
    
    @staticmethod
    def list_files(
        directory: Path,
        pattern: str = '*',
        recursive: bool = False
    ) -> list[Path]:
        """
        List files in directory.
        
        Args:
            directory: Directory to search
            pattern: Glob pattern (e.g., '*.json')
            recursive: Search recursively
            
        Returns:
            list: List of file paths
        """
        try:
            if not directory.exists():
                return []
            
            if recursive:
                files = list(directory.rglob(pattern))
            else:
                files = list(directory.glob(pattern))
            
            # Filter to only files (not directories)
            files = [f for f in files if f.is_file()]
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files in {directory}: {e}")
            return []
    
    @staticmethod
    def cleanup_old_files(
        directory: Path,
        max_age_seconds: int,
        pattern: str = '*'
    ) -> int:
        """
        Remove files older than specified age.
        
        Args:
            directory: Directory to clean
            max_age_seconds: Maximum file age in seconds
            pattern: Glob pattern for files to consider
            
        Returns:
            int: Number of files removed
        """
        try:
            if not directory.exists():
                return 0
            
            current_time = time.time()
            cutoff_time = current_time - max_age_seconds
            
            removed_count = 0
            
            for file_path in directory.glob(pattern):
                if not file_path.is_file():
                    continue
                
                # Check file age
                mtime = file_path.stat().st_mtime
                
                if mtime < cutoff_time:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove {file_path}: {e}")
            
            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} old files from {directory}"
                )
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Cleanup failed for {directory}: {e}")
            return 0
    
    @staticmethod
    def get_directory_size(directory: Path) -> int:
        """
        Calculate total size of directory.
        
        Args:
            directory: Directory path
            
        Returns:
            int: Total size in bytes
        """
        total_size = 0
        
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            
            return total_size
            
        except Exception as e:
            logger.error(f"Failed to calculate directory size: {e}")
            return 0


# Convenience functions for common operations

def atomic_write_json(file_path: Path, data: Any, indent: int = 2) -> None:
    """Convenience function for atomic JSON write."""
    FileUtils.atomic_write_json(file_path, data, indent)


def safe_read_json(file_path: Path, default: Any = None) -> Any:
    """Convenience function for safe JSON read."""
    return FileUtils.safe_read_json(file_path, default)


def safe_write_json(file_path: Path, data: Any, indent: int = 2) -> bool:
    """Convenience function for safe JSON write with locking."""
    return FileUtils.safe_write_json(file_path, data, indent)


@contextmanager
def file_lock(file_path: Path, timeout: float = 10.0) -> Generator[None, None, None]:
    """Convenience function for file locking."""
    with FileUtils.file_lock(file_path, timeout=timeout):
        yield
