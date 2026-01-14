#!/usr/bin/env python3
"""
Corruption Reporting System - Restore Script
Version: 1.0.0
Description: Restore system data from backup

This script restores:
- Submission files
- Evidence files
- Reports
- Validators configuration
- Hash chain
- Index

Usage:
    python scripts/restore.py --backup PATH [--force]
"""

import os
import sys
import json
import shutil
import tarfile
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

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

DATA_DIR = PROJECT_ROOT / 'backend' / 'data'
TEMP_DIR = PROJECT_ROOT / 'temp_restore'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_size(size_bytes: int) -> str:
    """
    Format bytes to human-readable size
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

# =============================================================================
# BACKUP EXTRACTION
# =============================================================================

def extract_backup(backup_path: Path, extract_dir: Path) -> Optional[Path]:
    """
    Extract backup archive
    
    Args:
        backup_path: Path to backup archive
        extract_dir: Directory to extract to
        
    Returns:
        Path to extracted backup directory or None if failed
    """
    try:
        if backup_path.suffix == '.gz' and backup_path.suffixes[-2:] == ['.tar', '.gz']:
            logger.info(f"Extracting archive: {backup_path.name}")
            
            with tarfile.open(backup_path, 'r:gz') as tar:
                tar.extractall(extract_dir)
            
            # Find extracted directory
            extracted_dirs = [d for d in extract_dir.iterdir() if d.is_dir()]
            if not extracted_dirs:
                logger.error("No directory found in archive")
                return None
            
            backup_dir = extracted_dirs[0]
            logger.info(f"  ✓ Extracted to: {backup_dir}")
            return backup_dir
        
        elif backup_path.is_dir():
            logger.info(f"Using directory backup: {backup_path}")
            return backup_path
        
        else:
            logger.error(f"Unknown backup format: {backup_path}")
            return None
    
    except Exception as e:
        logger.error(f"Failed to extract backup: {e}")
        return None

# =============================================================================
# MANIFEST OPERATIONS
# =============================================================================

def load_manifest(backup_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load backup manifest
    
    Args:
        backup_dir: Backup directory
        
    Returns:
        Manifest dictionary or None if failed
    """
    manifest_path = backup_dir / 'manifest.json'
    
    if not manifest_path.exists():
        logger.error("Manifest file not found in backup")
        return None
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        logger.info("Backup manifest loaded:")
        logger.info(f"  Version: {manifest.get('version', 'unknown')}")
        logger.info(f"  Created: {manifest.get('created_at', 'unknown')}")
        logger.info(f"  Files: {manifest['statistics']['total_files']}")
        logger.info(f"  Size: {format_size(manifest['statistics']['total_size'])}")
        
        return manifest
    
    except Exception as e:
        logger.error(f"Failed to load manifest: {e}")
        return None

# =============================================================================
# BACKUP VALIDATION
# =============================================================================

def validate_backup(backup_dir: Path, manifest: Dict[str, Any]) -> bool:
    """
    Validate backup before restoration
    
    Args:
        backup_dir: Backup directory
        manifest: Backup manifest
        
    Returns:
        True if valid
    """
    logger.info("\nValidating backup...")
    
    all_ok = True
    
    for component_name, component_info in manifest['components'].items():
        if component_info['backed_up']:
            component_path = backup_dir / component_name
            
            if not component_path.exists():
                logger.error(f"  ✗ Missing component: {component_name}")
                all_ok = False
            else:
                logger.debug(f"  ✓ Found component: {component_name}")
    
    if all_ok:
        logger.info("  ✓ Backup validation passed")
    else:
        logger.error("  ✗ Backup validation failed")
    
    return all_ok

# =============================================================================
# PRE-RESTORE BACKUP
# =============================================================================

def backup_current_data(backup_name: str = "pre_restore") -> bool:
    """
    Create backup of current data before restoration
    
    Args:
        backup_name: Name for pre-restore backup
        
    Returns:
        True if successful
    """
    logger.info("\nCreating pre-restore backup of current data...")
    
    if not DATA_DIR.exists():
        logger.info("  No existing data to backup")
        return True
    
    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_dir = PROJECT_ROOT / 'backups' / f"{backup_name}_{timestamp}"
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(DATA_DIR, backup_dir)
        
        logger.info(f"  ✓ Pre-restore backup created: {backup_dir}")
        return True
    
    except Exception as e:
        logger.warning(f"  ⚠ Failed to create pre-restore backup: {e}")
        return False

# =============================================================================
# RESTORE OPERATIONS
# =============================================================================

def restore_component(
    source_path: Path,
    dest_path: Path,
    component_name: str
) -> bool:
    """
    Restore a single component
    
    Args:
        source_path: Source path in backup
        dest_path: Destination path in data directory
        component_name: Name of component
        
    Returns:
        True if successful
    """
    try:
        # Remove existing if present
        if dest_path.exists():
            if dest_path.is_dir():
                shutil.rmtree(dest_path)
            else:
                dest_path.unlink()
        
        # Restore from backup
        if source_path.is_dir():
            shutil.copytree(source_path, dest_path, symlinks=False)
            file_count = sum(1 for _ in dest_path.rglob('*') if _.is_file())
            logger.info(f"  ✓ {component_name}: {file_count} files restored")
        else:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, dest_path)
            logger.info(f"  ✓ {component_name}: 1 file restored")
        
        return True
    
    except Exception as e:
        logger.error(f"  ✗ Failed to restore {component_name}: {e}")
        return False

def execute_restore(
    backup_dir: Path,
    manifest: Dict[str, Any],
    components: Optional[List[str]] = None
) -> bool:
    """
    Execute restoration from backup
    
    Args:
        backup_dir: Backup directory
        manifest: Backup manifest
        components: List of specific components to restore (None = all)
        
    Returns:
        True if successful
    """
    logger.info("\nRestoring components...")
    
    success_count = 0
    failure_count = 0
    
    for component_name, component_info in manifest['components'].items():
        # Skip if not backed up
        if not component_info['backed_up']:
            logger.debug(f"  - {component_name}: not in backup")
            continue
        
        # Skip if not in requested components
        if components and component_name not in components:
            logger.debug(f"  - {component_name}: skipped")
            continue
        
        # Determine paths
        source_path = backup_dir / component_name
        
        # Map component name to destination
        if component_name == 'submissions':
            dest_path = DATA_DIR / 'submissions'
        elif component_name == 'evidence':
            dest_path = DATA_DIR / 'evidence'
        elif component_name == 'reports':
            dest_path = DATA_DIR / 'reports'
        elif component_name == 'chain':
            dest_path = DATA_DIR / 'chain.json'
        elif component_name == 'validators':
            dest_path = DATA_DIR / 'validators.json'
        elif component_name == 'index':
            dest_path = DATA_DIR / 'index.json'
        else:
            logger.warning(f"  ⚠ Unknown component: {component_name}")
            continue
        
        # Restore component
        if restore_component(source_path, dest_path, component_name):
            success_count += 1
        else:
            failure_count += 1
    
    logger.info("")
    logger.info(f"Restoration complete: {success_count} succeeded, {failure_count} failed")
    
    return failure_count == 0

# =============================================================================
# POST-RESTORE VERIFICATION
# =============================================================================

def verify_restoration() -> bool:
    """
    Verify restoration was successful
    
    Returns:
        True if valid
    """
    logger.info("\nVerifying restoration...")
    
    all_ok = True
    
    # Check critical files
    critical_files = [
        DATA_DIR / 'chain.json',
        DATA_DIR / 'validators.json',
        DATA_DIR / 'index.json'
    ]
    
    for filepath in critical_files:
        if filepath.exists():
            # Try to load JSON
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.debug(f"  ✓ {filepath.name} valid")
            except Exception as e:
                logger.error(f"  ✗ {filepath.name} corrupted: {e}")
                all_ok = False
        else:
            logger.warning(f"  ⚠ {filepath.name} missing")
    
    # Check directories
    critical_dirs = [
        DATA_DIR / 'submissions',
        DATA_DIR / 'evidence'
    ]
    
    for dirpath in critical_dirs:
        if dirpath.exists() and dirpath.is_dir():
            logger.debug(f"  ✓ {dirpath.name}/ exists")
        else:
            logger.warning(f"  ⚠ {dirpath.name}/ missing")
    
    if all_ok:
        logger.info("  ✓ Restoration verification passed")
    
    return all_ok

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Restore data for Corruption Reporting System'
    )
    parser.add_argument(
        '--backup',
        type=str,
        required=True,
        help='Path to backup directory or archive'
    )
    parser.add_argument(
        '--components',
        type=str,
        nargs='+',
        default=None,
        help='Specific components to restore (default: all)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not create pre-restore backup'
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
    
    # Validate backup path
    backup_path = Path(args.backup)
    if not backup_path.exists():
        logger.error(f"Backup not found: {backup_path}")
        return 1
    
    # Print header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Restore")
    print("=" * 70)
    print("")
    
    # Show configuration
    logger.info("Configuration:")
    logger.info(f"  Backup: {backup_path}")
    logger.info(f"  Components: {args.components or 'all'}")
    logger.info(f"  Pre-restore backup: {not args.no_backup}")
    logger.info("")
    
    # Warn about data loss
    if not args.force:
        logger.warning("⚠ WARNING: This will overwrite existing data")
        try:
            response = input("Are you sure you want to continue? (yes/NO): ")
            if response.lower() != 'yes':
                logger.info("Restore cancelled by user")
                return 0
        except (KeyboardInterrupt, EOFError):
            logger.info("\nRestore cancelled by user")
            return 0
        print("")
    
    # Extract backup if needed
    logger.info("=" * 70)
    logger.info("Step 1: Extracting Backup")
    logger.info("=" * 70)
    
    if backup_path.suffix == '.gz':
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        backup_dir = extract_backup(backup_path, TEMP_DIR)
        cleanup_temp = True
    else:
        backup_dir = backup_path
        cleanup_temp = False
    
    if not backup_dir:
        logger.error("Failed to extract backup")
        return 1
    
    # Load manifest
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Loading Manifest")
    logger.info("=" * 70)
    
    manifest = load_manifest(backup_dir)
    if not manifest:
        logger.error("Failed to load manifest")
        if cleanup_temp:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return 1
    
    # Validate backup
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Validating Backup")
    logger.info("=" * 70)
    
    if not validate_backup(backup_dir, manifest):
        logger.error("Backup validation failed")
        if cleanup_temp:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        return 1
    
    # Create pre-restore backup
    if not args.no_backup:
        logger.info("\n" + "=" * 70)
        logger.info("Step 4: Pre-Restore Backup")
        logger.info("=" * 70)
        backup_current_data()
    
    # Execute restoration
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: Restoring Data")
    logger.info("=" * 70)
    
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    success = execute_restore(backup_dir, manifest, args.components)
    
    # Verify restoration
    logger.info("\n" + "=" * 70)
    logger.info("Step 6: Verification")
    logger.info("=" * 70)
    
    verify_restoration()
    
    # Cleanup
    if cleanup_temp:
        logger.info("\nCleaning up temporary files...")
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Restore Summary")
    logger.info("=" * 70)
    
    if success:
        logger.info("✓ Restoration completed successfully")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Verify data integrity")
        logger.info("  2. Start backend: cd backend && python -m uvicorn main:app --reload")
        logger.info("  3. Check system health: python scripts/health_check.py")
    else:
        logger.error("✗ Restoration completed with errors")
        logger.error("Some components may not have been restored correctly")
    
    logger.info("")
    
    return 0 if success else 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nRestore interrupted by user")
        # Cleanup temp directory if it exists
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
