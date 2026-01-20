#!/usr/bin/env python3
"""
Corruption Reporting System - Cleanup Script
Version: 1.0.0
Description: Remove old submissions and evidence files

This script:
- Removes submissions older than specified days
- Cleans up orphaned evidence files
- Updates hash chain and index
- Archives data before deletion (optional)
- Reports disk space freed

Usage:
    python scripts/cleanup.py [--days N] [--dry-run] [--archive]
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_RETENTION_DAYS = 90
MIN_RETENTION_DAYS = 7
DATA_DIR = PROJECT_ROOT / 'backend' / 'data'
SUBMISSIONS_DIR = DATA_DIR / 'submissions'
EVIDENCE_DIR = DATA_DIR / 'evidence'
REPORTS_DIR = DATA_DIR / 'reports'
CACHE_DIR = DATA_DIR / 'cache'
ARCHIVE_DIR = PROJECT_ROOT / 'archives'

# =============================================================================
# FILE OPERATIONS
# =============================================================================

def get_file_size(filepath: Path) -> int:
    """
    Get file size in bytes
    
    Args:
        filepath: Path to file
        
    Returns:
        File size in bytes
    """
    try:
        return filepath.stat().st_size
    except:
        return 0

def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def get_directory_size(directory: Path) -> int:
    """
    Get total size of directory
    
    Args:
        directory: Path to directory
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += get_file_size(item)
    except Exception as e:
        logger.warning(f"Error calculating directory size: {e}")
    
    return total_size

# =============================================================================
# SUBMISSION OPERATIONS
# =============================================================================

def load_submission(filepath: Path) -> Dict[str, Any]:
    """
    Load submission from JSON file
    
    Args:
        filepath: Path to submission JSON
        
    Returns:
        Submission data or None if error
    """
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load submission {filepath}: {e}")
        return None

def get_submission_age(submission: Dict[str, Any]) -> int:
    """
    Get submission age in days
    
    Args:
        submission: Submission data
        
    Returns:
        Age in days
    """
    try:
        timestamp_str = submission.get('timestamp') or submission.get('created_at')
        if not timestamp_str:
            return 0
        
        # Parse timestamp (handle both formats)
        if timestamp_str.endswith('Z'):
            timestamp_str = timestamp_str[:-1]
        
        created_at = datetime.fromisoformat(timestamp_str)
        age = datetime.utcnow() - created_at
        return age.days
    
    except Exception as e:
        logger.warning(f"Failed to calculate submission age: {e}")
        return 0

def find_old_submissions(retention_days: int) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Find submissions older than retention period
    
    Args:
        retention_days: Retention period in days
        
    Returns:
        List of (filepath, submission_data) tuples
    """
    logger.info(f"Scanning for submissions older than {retention_days} days...")
    
    if not SUBMISSIONS_DIR.exists():
        logger.warning(f"Submissions directory not found: {SUBMISSIONS_DIR}")
        return []
    
    old_submissions = []
    total_submissions = 0
    
    for submission_file in SUBMISSIONS_DIR.glob('*.json'):
        total_submissions += 1
        submission = load_submission(submission_file)
        
        if submission:
            age_days = get_submission_age(submission)
            if age_days > retention_days:
                old_submissions.append((submission_file, submission))
    
    logger.info(f"Found {len(old_submissions)} old submissions out of {total_submissions} total")
    
    return old_submissions

# =============================================================================
# EVIDENCE OPERATIONS
# =============================================================================

def find_orphaned_evidence(valid_submission_ids: set) -> List[Path]:
    """
    Find evidence files not linked to any submission
    
    Args:
        valid_submission_ids: Set of valid submission IDs
        
    Returns:
        List of orphaned evidence file paths
    """
    logger.info("Scanning for orphaned evidence files...")
    
    if not EVIDENCE_DIR.exists():
        logger.warning(f"Evidence directory not found: {EVIDENCE_DIR}")
        return []
    
    orphaned_files = []
    total_files = 0
    
    for evidence_file in EVIDENCE_DIR.rglob('*'):
        if evidence_file.is_file() and evidence_file.name != '.gitkeep':
            total_files += 1
            
            # Extract submission ID from filename or path
            # Evidence files are typically stored as: evidence/YYYY/MM/submission_id_filename
            parts = evidence_file.parts
            
            # Check if submission ID is in valid set
            is_valid = False
            for part in parts:
                if part in valid_submission_ids:
                    is_valid = True
                    break
            
            if not is_valid:
                # Check filename
                filename_parts = evidence_file.stem.split('_')
                if filename_parts and filename_parts[0] in valid_submission_ids:
                    is_valid = True
            
            if not is_valid:
                orphaned_files.append(evidence_file)
    
    logger.info(f"Found {len(orphaned_files)} orphaned files out of {total_files} total")
    
    return orphaned_files

# =============================================================================
# CACHE OPERATIONS
# =============================================================================

def cleanup_cache() -> int:
    """
    Clean up cache directory
    
    Returns:
        Number of bytes freed
    """
    logger.info("Cleaning cache directory...")
    
    if not CACHE_DIR.exists():
        logger.info("No cache directory found")
        return 0
    
    size_before = get_directory_size(CACHE_DIR)
    
    # Remove all cache files except .gitkeep
    for cache_file in CACHE_DIR.rglob('*'):
        if cache_file.is_file() and cache_file.name != '.gitkeep':
            try:
                cache_file.unlink()
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")
    
    size_freed = size_before
    logger.info(f"Freed {format_size(size_freed)} from cache")
    
    return size_freed

# =============================================================================
# ARCHIVE OPERATIONS
# =============================================================================

def archive_submission(submission_path: Path, submission_data: Dict[str, Any]) -> bool:
    """
    Archive submission before deletion
    
    Args:
        submission_path: Path to submission file
        submission_data: Submission data
        
    Returns:
        True if successful
    """
    try:
        # Create archive directory with date
        archive_date = datetime.utcnow().strftime('%Y-%m-%d')
        archive_path = ARCHIVE_DIR / archive_date
        archive_path.mkdir(parents=True, exist_ok=True)
        
        # Copy submission file
        dest_path = archive_path / submission_path.name
        shutil.copy2(submission_path, dest_path)
        
        return True
    
    except Exception as e:
        logger.warning(f"Failed to archive submission {submission_path}: {e}")
        return False

# =============================================================================
# INDEX OPERATIONS
# =============================================================================

def update_index(removed_submission_ids: List[str]) -> bool:
    """
    Update index after removing submissions
    
    Args:
        removed_submission_ids: List of removed submission IDs
        
    Returns:
        True if successful
    """
    index_path = DATA_DIR / 'index.json'
    
    if not index_path.exists():
        logger.warning("Index file not found")
        return True  # Not critical
    
    try:
        # Load index
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        # Remove deleted submissions
        submissions = index_data.get('submissions', {})
        for submission_id in removed_submission_ids:
            if submission_id in submissions:
                del submissions[submission_id]
        
        # Update metadata
        index_data['metadata']['total_submissions'] = len(submissions)
        index_data['metadata']['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        
        # Save index
        with open(index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"Updated index (removed {len(removed_submission_ids)} entries)")
        return True
    
    except Exception as e:
        logger.error(f"Failed to update index: {e}")
        return False

# =============================================================================
# CLEANUP EXECUTION
# =============================================================================

def execute_cleanup(
    retention_days: int,
    dry_run: bool = False,
    archive: bool = False,
    clean_cache: bool = True
) -> Dict[str, Any]:
    """
    Execute cleanup operation
    
    Args:
        retention_days: Retention period in days
        dry_run: If True, only report what would be deleted
        archive: If True, archive submissions before deletion
        clean_cache: If True, clean cache directory
        
    Returns:
        Dictionary with cleanup statistics
    """
    stats = {
        'submissions_deleted': 0,
        'evidence_deleted': 0,
        'files_deleted': 0,
        'bytes_freed': 0,
        'archived': 0
    }
    
    logger.info("\n" + "=" * 70)
    if dry_run:
        logger.info("DRY RUN - No files will be deleted")
    else:
        logger.info("CLEANUP - Files will be deleted")
    logger.info("=" * 70)
    logger.info("")
    
    # Find old submissions
    old_submissions = find_old_submissions(retention_days)
    
    if not old_submissions and not clean_cache:
        logger.info("No old submissions found")
        return stats
    
    # Get current valid submission IDs
    valid_submission_ids = set()
    if SUBMISSIONS_DIR.exists():
        for submission_file in SUBMISSIONS_DIR.glob('*.json'):
            submission = load_submission(submission_file)
            if submission:
                submission_id = submission.get('submission_id') or submission.get('pseudonym')
                if submission_id:
                    valid_submission_ids.add(submission_id)
    
    # Remove old submissions from valid set
    removed_ids = []
    for submission_path, submission in old_submissions:
        submission_id = submission.get('submission_id') or submission.get('pseudonym')
        if submission_id and submission_id in valid_submission_ids:
            valid_submission_ids.remove(submission_id)
            removed_ids.append(submission_id)
    
    # Find orphaned evidence
    orphaned_evidence = find_orphaned_evidence(valid_submission_ids)
    
    # Archive if requested
    if archive and not dry_run:
        logger.info("\nArchiving submissions...")
        for submission_path, submission in old_submissions:
            if archive_submission(submission_path, submission):
                stats['archived'] += 1
    
    # Delete old submissions
    if old_submissions:
        logger.info(f"\nRemoving {len(old_submissions)} old submissions...")
        
        for submission_path, submission in old_submissions:
            size = get_file_size(submission_path)
            
            if not dry_run:
                try:
                    submission_path.unlink()
                    stats['submissions_deleted'] += 1
                    stats['files_deleted'] += 1
                    stats['bytes_freed'] += size
                    logger.debug(f"  Deleted: {submission_path.name}")
                except Exception as e:
                    logger.warning(f"  Failed to delete {submission_path}: {e}")
            else:
                stats['submissions_deleted'] += 1
                stats['files_deleted'] += 1
                stats['bytes_freed'] += size
                logger.debug(f"  Would delete: {submission_path.name} ({format_size(size)})")
    
    # Delete orphaned evidence
    if orphaned_evidence:
        logger.info(f"\nRemoving {len(orphaned_evidence)} orphaned evidence files...")
        
        for evidence_path in orphaned_evidence:
            size = get_file_size(evidence_path)
            
            if not dry_run:
                try:
                    evidence_path.unlink()
                    stats['evidence_deleted'] += 1
                    stats['files_deleted'] += 1
                    stats['bytes_freed'] += size
                    logger.debug(f"  Deleted: {evidence_path.name}")
                except Exception as e:
                    logger.warning(f"  Failed to delete {evidence_path}: {e}")
            else:
                stats['evidence_deleted'] += 1
                stats['files_deleted'] += 1
                stats['bytes_freed'] += size
                logger.debug(f"  Would delete: {evidence_path.name} ({format_size(size)})")
    
    # Clean cache
    if clean_cache:
        if not dry_run:
            cache_freed = cleanup_cache()
            stats['bytes_freed'] += cache_freed
        else:
            cache_size = get_directory_size(CACHE_DIR)
            logger.info(f"Would free {format_size(cache_size)} from cache")
            stats['bytes_freed'] += cache_size
    
    # Update index
    if not dry_run and removed_ids:
        update_index(removed_ids)
    
    return stats

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Cleanup old submissions for Corruption Reporting System'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=DEFAULT_RETENTION_DAYS,
        help=f'Retention period in days (default: {DEFAULT_RETENTION_DAYS})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    parser.add_argument(
        '--archive',
        action='store_true',
        help='Archive submissions before deletion'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Do not clean cache directory'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Validate retention days
    if args.days < MIN_RETENTION_DAYS:
        logger.error(f"Retention period must be at least {MIN_RETENTION_DAYS} days")
        return 1
    
    # Print header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Cleanup")
    print("=" * 70)
    print("")
    
    # Show configuration
    logger.info("Configuration:")
    logger.info(f"  Retention period: {args.days} days")
    logger.info(f"  Dry run: {args.dry_run}")
    logger.info(f"  Archive: {args.archive}")
    logger.info(f"  Clean cache: {not args.no_cache}")
    logger.info("")
    
    # Confirm unless dry-run or force
    if not args.dry_run and not args.force:
        logger.warning("⚠ WARNING: This will permanently delete data")
        try:
            response = input("Are you sure you want to continue? (yes/NO): ")
            if response.lower() != 'yes':
                logger.info("Cleanup cancelled by user")
                return 0
        except (KeyboardInterrupt, EOFError):
            logger.info("\nCleanup cancelled by user")
            return 0
        print("")
    
    # Execute cleanup
    stats = execute_cleanup(
        retention_days=args.days,
        dry_run=args.dry_run,
        archive=args.archive,
        clean_cache=not args.no_cache
    )
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Cleanup Summary")
    logger.info("=" * 70)
    logger.info(f"Submissions deleted: {stats['submissions_deleted']}")
    logger.info(f"Evidence files deleted: {stats['evidence_deleted']}")
    logger.info(f"Total files deleted: {stats['files_deleted']}")
    logger.info(f"Disk space freed: {format_size(stats['bytes_freed'])}")
    
    if args.archive:
        logger.info(f"Submissions archived: {stats['archived']}")
    
    if args.dry_run:
        logger.info("\n Dry run complete (no files were deleted)")
    else:
        logger.info("\n Cleanup complete")
    
    logger.info("")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nCleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
