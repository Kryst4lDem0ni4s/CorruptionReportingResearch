#!/usr/bin/env python3
"""
Corruption Reporting System - Backup Script
Version: 1.0.0
Description: Backup system data to archive

This script backs up:
- Submission files
- Evidence files
- Reports
- Validators configuration
- Hash chain
- Index

Usage:
    python scripts/backup.py [--output PATH] [--compress]
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
from typing import Dict, Any, List, Optional

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
DEFAULT_BACKUP_DIR = PROJECT_ROOT / 'backups'

BACKUP_COMPONENTS = {
    'submissions': DATA_DIR / 'submissions',
    'evidence': DATA_DIR / 'evidence',
    'reports': DATA_DIR / 'reports',
    'chain': DATA_DIR / 'chain.json',
    'validators': DATA_DIR / 'validators.json',
    'index': DATA_DIR / 'index.json'
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_directory_size(directory: Path) -> int:
    """
    Calculate total size of directory
    
    Args:
        directory: Path to directory
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for item in directory.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
    except Exception as e:
        logger.warning(f"Error calculating directory size: {e}")
    
    return total_size

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

def generate_backup_name() -> str:
    """
    Generate timestamped backup name
    
    Returns:
        Backup name string
    """
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    return f"backup_{timestamp}"

# =============================================================================
# BACKUP MANIFEST
# =============================================================================

def create_manifest(backup_components: Dict[str, Path]) -> Dict[str, Any]:
    """
    Create backup manifest with metadata
    
    Args:
        backup_components: Dictionary of components backed up
        
    Returns:
        Manifest dictionary
    """
    manifest = {
        'version': '1.0.0',
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'backup_type': 'full',
        'components': {},
        'statistics': {
            'total_files': 0,
            'total_size': 0
        }
    }
    
    total_files = 0
    total_size = 0
    
    for component_name, component_path in backup_components.items():
        component_info = {
            'backed_up': False,
            'path': str(component_path),
            'type': 'directory' if component_path.is_dir() else 'file',
            'files': 0,
            'size': 0
        }
        
        if component_path.exists():
            component_info['backed_up'] = True
            
            if component_path.is_dir():
                files = list(component_path.rglob('*'))
                component_info['files'] = sum(1 for f in files if f.is_file())
                component_info['size'] = get_directory_size(component_path)
            else:
                component_info['files'] = 1
                component_info['size'] = component_path.stat().st_size
            
            total_files += component_info['files']
            total_size += component_info['size']
        
        manifest['components'][component_name] = component_info
    
    manifest['statistics']['total_files'] = total_files
    manifest['statistics']['total_size'] = total_size
    
    return manifest

# =============================================================================
# BACKUP OPERATIONS
# =============================================================================

def backup_component(
    component_path: Path,
    backup_dir: Path,
    component_name: str
) -> bool:
    """
    Backup a single component
    
    Args:
        component_path: Path to component
        backup_dir: Backup destination directory
        component_name: Name of component
        
    Returns:
        True if successful
    """
    if not component_path.exists():
        logger.warning(f"Component not found: {component_name}")
        return False
    
    try:
        dest_path = backup_dir / component_name
        
        if component_path.is_dir():
            # Copy directory
            shutil.copytree(component_path, dest_path, symlinks=False)
            file_count = sum(1 for _ in dest_path.rglob('*') if _.is_file())
            logger.info(f"   {component_name}: {file_count} files")
        else:
            # Copy file
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(component_path, dest_path)
            logger.info(f"   {component_name}: 1 file")
        
        return True
    
    except Exception as e:
        logger.error(f"   Failed to backup {component_name}: {e}")
        return False

def create_compressed_archive(
    backup_dir: Path,
    output_path: Path
) -> bool:
    """
    Create compressed tar.gz archive
    
    Args:
        backup_dir: Directory to compress
        output_path: Output archive path
        
    Returns:
        True if successful
    """
    try:
        logger.info(f"Creating compressed archive: {output_path.name}")
        
        with tarfile.open(output_path, 'w:gz') as tar:
            tar.add(backup_dir, arcname=backup_dir.name)
        
        archive_size = output_path.stat().st_size
        logger.info(f"  Archive size: {format_size(archive_size)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        return False

def verify_backup(backup_dir: Path) -> bool:
    """
    Verify backup integrity
    
    Args:
        backup_dir: Backup directory to verify
        
    Returns:
        True if valid
    """
    logger.info("Verifying backup integrity...")
    
    # Check manifest
    manifest_path = backup_dir / 'manifest.json'
    if not manifest_path.exists():
        logger.error("   Manifest file missing")
        return False
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        logger.info(f"   Manifest valid")
        
        # Verify components
        all_ok = True
        for component_name, component_info in manifest['components'].items():
            if component_info['backed_up']:
                component_path = backup_dir / component_name
                if not component_path.exists():
                    logger.error(f"   Component missing: {component_name}")
                    all_ok = False
                else:
                    logger.debug(f"   Component present: {component_name}")
        
        if all_ok:
            logger.info("   All components verified")
        
        return all_ok
    
    except Exception as e:
        logger.error(f"   Verification failed: {e}")
        return False

# =============================================================================
# MAIN BACKUP FUNCTION
# =============================================================================

def execute_backup(
    output_dir: Path,
    compress: bool = True,
    verify: bool = True
) -> Optional[Path]:
    """
    Execute full backup
    
    Args:
        output_dir: Output directory for backup
        compress: Whether to create compressed archive
        verify: Whether to verify backup
        
    Returns:
        Path to backup (directory or archive) or None if failed
    """
    logger.info("\n" + "=" * 70)
    logger.info("Starting Backup")
    logger.info("=" * 70)
    
    # Create backup directory
    backup_name = generate_backup_name()
    backup_dir = output_dir / backup_name
    
    try:
        backup_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Backup location: {backup_dir}")
        logger.info("")
    except Exception as e:
        logger.error(f"Failed to create backup directory: {e}")
        return None
    
    # Backup components
    logger.info("Backing up components...")
    
    backed_up_components = {}
    for component_name, component_path in BACKUP_COMPONENTS.items():
        if backup_component(component_path, backup_dir, component_name):
            backed_up_components[component_name] = component_path
    
    logger.info("")
    
    if not backed_up_components:
        logger.error("No components were backed up")
        shutil.rmtree(backup_dir, ignore_errors=True)
        return None
    
    # Create manifest
    logger.info("Creating manifest...")
    manifest = create_manifest(backed_up_components)
    
    manifest_path = backup_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    logger.info(f"   Manifest created")
    logger.info("")
    
    # Verify backup
    if verify:
        if not verify_backup(backup_dir):
            logger.error("Backup verification failed")
            return None
        logger.info("")
    
    # Compress if requested
    final_path = backup_dir
    if compress:
        archive_path = output_dir / f"{backup_name}.tar.gz"
        if create_compressed_archive(backup_dir, archive_path):
            # Remove uncompressed directory
            shutil.rmtree(backup_dir)
            final_path = archive_path
            logger.info("")
        else:
            logger.warning("Compression failed, keeping uncompressed backup")
    
    return final_path

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Backup data for Corruption Reporting System'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help=f'Output directory for backup (default: {DEFAULT_BACKUP_DIR})'
    )
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Do not compress backup (keep as directory)'
    )
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip backup verification'
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
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = DEFAULT_BACKUP_DIR
    
    # Print header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Backup")
    print("=" * 70)
    print("")
    
    # Show configuration
    logger.info("Configuration:")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Compression: {not args.no_compress}")
    logger.info(f"  Verification: {not args.no_verify}")
    logger.info("")
    
    # Check if data directory exists
    if not DATA_DIR.exists():
        logger.error(f"Data directory not found: {DATA_DIR}")
        logger.error("Please run initialize_storage.py first")
        return 1
    
    # Execute backup
    backup_path = execute_backup(
        output_dir=output_dir,
        compress=not args.no_compress,
        verify=not args.no_verify
    )
    
    if not backup_path:
        logger.error("Backup failed")
        return 1
    
    # Print summary
    logger.info("=" * 70)
    logger.info("Backup Summary")
    logger.info("=" * 70)
    
    backup_size = backup_path.stat().st_size
    logger.info(f"Backup location: {backup_path}")
    logger.info(f"Backup size: {format_size(backup_size)}")
    
    # Load and display manifest
    if backup_path.suffix == '.gz':
        logger.info("\nTo view backup contents:")
        logger.info(f"  tar -tzf {backup_path}")
    else:
        manifest_path = backup_path / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
            
            logger.info(f"\nBackup contains:")
            logger.info(f"  Total files: {manifest['statistics']['total_files']}")
            logger.info(f"  Total size: {format_size(manifest['statistics']['total_size'])}")
    
    logger.info("\n Backup completed successfully")
    logger.info("")
    logger.info(f"To restore this backup, run:")
    logger.info(f"  python scripts/restore.py --backup {backup_path}")
    logger.info("")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nBackup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
