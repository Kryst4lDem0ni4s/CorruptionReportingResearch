#!/usr/bin/env python3
"""
Corruption Reporting System - Data Migration Script
Version: 1.0.0
Description: Migrate data between versions

This script handles:
- Schema migrations
- Data format updates
- Version upgrades
- Automatic backups
- Rollback support

Usage:
    python scripts/migrate.py [--target-version X.Y.Z] [--dry-run]
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable

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
BACKUP_DIR = PROJECT_ROOT / 'backups'

CURRENT_VERSION = '1.0.0'

# =============================================================================
# VERSION UTILITIES
# =============================================================================

def parse_version(version_str: str) -> tuple:
    """Parse version string to tuple"""
    try:
        return tuple(map(int, version_str.split('.')))
    except:
        return (0, 0, 0)

def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings
    
    Returns:
        -1 if v1 < v2, 0 if v1 == v2, 1 if v1 > v2
    """
    v1_tuple = parse_version(v1)
    v2_tuple = parse_version(v2)
    
    if v1_tuple < v2_tuple:
        return -1
    elif v1_tuple > v2_tuple:
        return 1
    else:
        return 0

def detect_current_version() -> str:
    """
    Detect current data version
    
    Returns:
        Version string
    """
    # Check chain.json for version
    chain_file = DATA_DIR / 'chain.json'
    
    if chain_file.exists():
        try:
            with open(chain_file, 'r') as f:
                data = json.load(f)
            return data.get('version', '1.0.0')
        except:
            pass
    
    return '1.0.0'

# =============================================================================
# BACKUP OPERATIONS
# =============================================================================

def create_migration_backup() -> Optional[Path]:
    """
    Create backup before migration
    
    Returns:
        Path to backup or None if failed
    """
    logger.info("Creating pre-migration backup...")
    
    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_name = f"pre_migration_{timestamp}"
        backup_path = BACKUP_DIR / backup_name
        
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy data directory
        shutil.copytree(DATA_DIR, backup_path / 'data', symlinks=False)
        
        # Create backup manifest
        manifest = {
            'backup_type': 'pre_migration',
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'data_version': detect_current_version()
        }
        
        with open(backup_path / 'manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"  ✓ Backup created: {backup_path}")
        return backup_path
    
    except Exception as e:
        logger.error(f"  ✗ Backup failed: {e}")
        return None

# =============================================================================
# MIGRATION FUNCTIONS
# =============================================================================

def migrate_chain_1_0_to_1_1(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate chain.json from 1.0 to 1.1
    
    Changes:
    - Add block hash field
    - Add merkle root
    """
    logger.info("  Migrating chain structure...")
    
    # Update version
    data['version'] = '1.1.0'
    
    # Add new fields to each block
    for block in data.get('chain', []):
        if 'block_hash' not in block:
            block['block_hash'] = block.get('data_hash', '0' * 64)
        if 'merkle_root' not in block:
            block['merkle_root'] = block.get('data_hash', '0' * 64)
    
    return data

def migrate_validators_1_0_to_1_1(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate validators.json from 1.0 to 1.1
    
    Changes:
    - Add validator reputation field
    - Add last_active timestamp
    """
    logger.info("  Migrating validator structure...")
    
    # Update version
    data['version'] = '1.1.0'
    
    # Add new fields to each validator
    for validator in data.get('validators', []):
        if 'reputation' not in validator:
            validator['reputation'] = validator.get('reliability_score', 0.9)
        if 'last_active' not in validator:
            validator['last_active'] = validator.get('created_at', datetime.utcnow().isoformat() + 'Z')
    
    return data

def migrate_index_1_0_to_1_1(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate index.json from 1.0 to 1.1
    
    Changes:
    - Add search index field
    - Add tags support
    """
    logger.info("  Migrating index structure...")
    
    # Update version
    data['version'] = '1.1.0'
    
    # Add search index metadata
    if 'search_index' not in data.get('metadata', {}):
        data['metadata']['search_index'] = {
            'enabled': False,
            'last_updated': datetime.utcnow().isoformat() + 'Z'
        }
    
    return data

# =============================================================================
# MIGRATION REGISTRY
# =============================================================================

# Define migration paths
# Key: (from_version, to_version), Value: list of migration functions
MIGRATIONS: Dict[tuple, List[Callable]] = {
    ('1.0.0', '1.1.0'): [
        ('chain.json', migrate_chain_1_0_to_1_1),
        ('validators.json', migrate_validators_1_0_to_1_1),
        ('index.json', migrate_index_1_0_to_1_1)
    ]
}

def get_migration_path(from_version: str, to_version: str) -> List[tuple]:
    """
    Get migration path from one version to another
    
    Returns:
        List of (version_pair, migrations) tuples
    """
    # For now, only direct migrations supported
    key = (from_version, to_version)
    
    if key in MIGRATIONS:
        return [(key, MIGRATIONS[key])]
    
    return []

# =============================================================================
# MIGRATION EXECUTION
# =============================================================================

def apply_migration(
    file_path: Path,
    migration_func: Callable,
    dry_run: bool = False
) -> bool:
    """
    Apply a single migration to a file
    
    Args:
        file_path: Path to file
        migration_func: Migration function
        dry_run: If True, don't write changes
        
    Returns:
        True if successful
    """
    try:
        # Load file
        if not file_path.exists():
            logger.warning(f"    File not found: {file_path.name}")
            return True  # Not critical
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Apply migration
        migrated_data = migration_func(data)
        
        # Save if not dry run
        if not dry_run:
            with open(file_path, 'w') as f:
                json.dump(migrated_data, f, indent=2)
            logger.info(f"    ✓ {file_path.name} migrated")
        else:
            logger.info(f"    ✓ {file_path.name} would be migrated (dry run)")
        
        return True
    
    except Exception as e:
        logger.error(f"    ✗ {file_path.name} migration failed: {e}")
        return False

def execute_migration(
    from_version: str,
    to_version: str,
    dry_run: bool = False
) -> bool:
    """
    Execute migration from one version to another
    
    Args:
        from_version: Source version
        to_version: Target version
        dry_run: If True, simulate migration
        
    Returns:
        True if successful
    """
    logger.info(f"\nMigrating from v{from_version} to v{to_version}")
    logger.info("=" * 70)
    
    # Get migration path
    migration_path = get_migration_path(from_version, to_version)
    
    if not migration_path:
        logger.warning("No migration path found")
        logger.info("Data is already up to date or migration not defined")
        return True
    
    # Execute migrations
    all_success = True
    
    for version_pair, migrations in migration_path:
        logger.info(f"\nApplying migrations for {version_pair[0]} -> {version_pair[1]}:")
        
        for file_name, migration_func in migrations:
            file_path = DATA_DIR / file_name
            success = apply_migration(file_path, migration_func, dry_run)
            
            if not success:
                all_success = False
    
    return all_success

# =============================================================================
# VALIDATION
# =============================================================================

def validate_migrated_data() -> bool:
    """
    Validate data after migration
    
    Returns:
        True if valid
    """
    logger.info("\nValidating migrated data...")
    
    all_ok = True
    
    # Check critical files
    critical_files = [
        DATA_DIR / 'chain.json',
        DATA_DIR / 'validators.json',
        DATA_DIR / 'index.json'
    ]
    
    for file_path in critical_files:
        if not file_path.exists():
            logger.error(f"  ✗ {file_path.name} missing")
            all_ok = False
            continue
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"  ✓ {file_path.name} valid")
        except Exception as e:
            logger.error(f"  ✗ {file_path.name} corrupted: {e}")
            all_ok = False
    
    if all_ok:
        logger.info("  ✓ All validations passed")
    
    return all_ok

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Migrate data for Corruption Reporting System'
    )
    parser.add_argument(
        '--target-version',
        type=str,
        default=CURRENT_VERSION,
        help=f'Target version (default: {CURRENT_VERSION})'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate migration without making changes'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip pre-migration backup'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Data Migration")
    print("=" * 70)
    print("")
    
    # Detect current version
    current_version = detect_current_version()
    target_version = args.target_version
    
    logger.info(f"Current data version: {current_version}")
    logger.info(f"Target version: {target_version}")
    logger.info("")
    
    # Check if migration needed
    version_cmp = compare_versions(current_version, target_version)
    
    if version_cmp == 0:
        logger.info("✓ Data is already at target version")
        return 0
    elif version_cmp > 0:
        logger.warning("⚠ Current version is newer than target version")
        logger.warning("Downgrade not supported")
        return 1
    
    # Confirm migration
    if not args.force and not args.dry_run:
        logger.warning("⚠ WARNING: This will modify your data")
        try:
            response = input("Continue with migration? (yes/NO): ")
            if response.lower() != 'yes':
                logger.info("Migration cancelled")
                return 0
        except (KeyboardInterrupt, EOFError):
            logger.info("\nMigration cancelled")
            return 0
        print("")
    
    # Create backup
    if not args.no_backup and not args.dry_run:
        backup_path = create_migration_backup()
        if not backup_path:
            logger.error("Failed to create backup")
            logger.error("Migration aborted for safety")
            return 1
        logger.info("")
    
    # Execute migration
    success = execute_migration(
        from_version=current_version,
        to_version=target_version,
        dry_run=args.dry_run
    )
    
    if not success:
        logger.error("\n✗ Migration failed")
        logger.error("Please check errors above")
        
        if not args.no_backup and not args.dry_run:
            logger.info(f"\nTo restore from backup:")
            logger.info(f"  python scripts/restore.py --backup {backup_path}")
        
        return 1
    
    # Validate if not dry run
    if not args.dry_run:
        if not validate_migrated_data():
            logger.error("\n✗ Validation failed")
            return 1
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Migration Summary")
    logger.info("=" * 70)
    
    if args.dry_run:
        logger.info("✓ Dry run completed successfully")
        logger.info("\nNo changes were made")
        logger.info("Run without --dry-run to apply migration")
    else:
        logger.info("✓ Migration completed successfully")
        logger.info(f"\nData migrated: {current_version} → {target_version}")
        
        if not args.no_backup:
            logger.info(f"Backup location: {backup_path}")
    
    logger.info("")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nMigration interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
