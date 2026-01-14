# """
# Initialize Storage - Create data directory structure

# Creates all necessary directories and .gitkeep files for the data storage system.
# """

# import logging
# from pathlib import Path
# from datetime import datetime

# # Initialize logger
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def create_directory_structure(base_dir: Path = Path('backend/data')):
#     """
#     Create complete data directory structure.
    
#     Args:
#         base_dir: Base data directory path
#     """
#     logger.info(f"Creating data directory structure at: {base_dir}")
    
#     # Define directory structure
#     directories = [
#         # Root data directory
#         base_dir,
        
#         # Submissions directory
#         base_dir / 'submissions',
        
#         # Evidence directory with year/month sharding
#         base_dir / 'evidence',
#         base_dir / 'evidence' / '2026',
#         base_dir / 'evidence' / '2026' / '01',
#         base_dir / 'evidence' / '2026' / '02',
#         base_dir / 'evidence' / '2026' / '03',
#         base_dir / 'evidence' / '2026' / '04',
#         base_dir / 'evidence' / '2026' / '05',
#         base_dir / 'evidence' / '2026' / '06',
#         base_dir / 'evidence' / '2026' / '07',
#         base_dir / 'evidence' / '2026' / '08',
#         base_dir / 'evidence' / '2026' / '09',
#         base_dir / 'evidence' / '2026' / '10',
#         base_dir / 'evidence' / '2026' / '11',
#         base_dir / 'evidence' / '2026' / '12',
        
#         # Reports directory
#         base_dir / 'reports',
        
#         # Cache directory
#         base_dir / 'cache',
        
#         # Archive directory (for old reports)
#         base_dir / 'reports_archive',
#     ]
    
#     # Create directories
#     created_count = 0
#     for directory in directories:
#         if not directory.exists():
#             directory.mkdir(parents=True, exist_ok=True)
#             logger.info(f"Created directory: {directory}")
#             created_count += 1
            
#             # Create .gitkeep file to track empty directories in Git
#             gitkeep_file = directory / '.gitkeep'
#             if not gitkeep_file.exists():
#                 gitkeep_file.touch()
#                 logger.debug(f"Created .gitkeep: {gitkeep_file}")
#         else:
#             logger.debug(f"Directory already exists: {directory}")
    
#     # Create initial JSON files if they don't exist
#     json_files = {
#         'chain.json': {
#             'genesis_block': {
#                 'block_number': 0,
#                 'timestamp': datetime.now().isoformat(),
#                 'previous_hash': '0' * 64,
#                 'data': 'Genesis Block',
#                 'hash': None  # Will be computed on first use
#             },
#             'blocks': [],
#             'created_at': datetime.now().isoformat()
#         },
#         'validators.json': {
#             'validators': [],
#             'created_at': datetime.now().isoformat(),
#             'last_updated': datetime.now().isoformat()
#         },
#         'index.json': {
#             'submissions': {},
#             'created_at': datetime.now().isoformat(),
#             'last_updated': datetime.now().isoformat()
#         }
#     }
    
#     import json
    
#     for filename, initial_data in json_files.items():
#         file_path = base_dir / filename
#         if not file_path.exists():
#             with open(file_path, 'w') as f:
#                 json.dump(initial_data, f, indent=2)
#             logger.info(f"Created initial file: {file_path}")
#         else:
#             logger.debug(f"File already exists: {file_path}")
    
#     logger.info(
#         f"Data directory structure initialized successfully! "
#         f"Created {created_count} new directories."
#     )
    
#     return base_dir


# def add_future_months(base_dir: Path = Path('backend/data'), year: int = 2026):
#     """
#     Add month directories for the rest of the year.
    
#     Args:
#         base_dir: Base data directory
#         year: Year for month directories
#     """
#     evidence_dir = base_dir / 'evidence' / str(year)
    
#     for month in range(1, 13):
#         month_dir = evidence_dir / f'{month:02d}'
#         if not month_dir.exists():
#             month_dir.mkdir(parents=True, exist_ok=True)
#             (month_dir / '.gitkeep').touch()
#             logger.info(f"Created month directory: {month_dir}")


# def verify_structure(base_dir: Path = Path('backend/data')) -> bool:
#     """
#     Verify data directory structure is complete.
    
#     Args:
#         base_dir: Base data directory
        
#     Returns:
#         bool: True if structure is valid
#     """
#     logger.info("Verifying data directory structure...")
    
#     required_dirs = [
#         base_dir / 'submissions',
#         base_dir / 'evidence',
#         base_dir / 'reports',
#         base_dir / 'cache',
#     ]
    
#     required_files = [
#         base_dir / 'chain.json',
#         base_dir / 'validators.json',
#         base_dir / 'index.json',
#     ]
    
#     all_valid = True
    
#     # Check directories
#     for directory in required_dirs:
#         if not directory.exists():
#             logger.error(f"Missing directory: {directory}")
#             all_valid = False
#         else:
#             logger.debug(f"✓ Directory exists: {directory}")
    
#     # Check files
#     for file_path in required_files:
#         if not file_path.exists():
#             logger.error(f"Missing file: {file_path}")
#             all_valid = False
#         else:
#             logger.debug(f"✓ File exists: {file_path}")
    
#     if all_valid:
#         logger.info("✓ Data directory structure is valid!")
#     else:
#         logger.error("✗ Data directory structure has issues!")
    
#     return all_valid

# def create_initial_json_files(base_dir: Path) -> None:
#     """
#     Create initial JSON files with proper structure.
    
#     Args:
#         base_dir: Base data directory
#     """
#     from datetime import datetime
#     import json
    
#     logger.info("Creating initial JSON files...")
    
#     current_time = datetime.now().isoformat()
    
#     # 1. Chain JSON - Hash chain state
#     chain_data = {
#         "version": "1.0.0",
#         "created_at": current_time,
#         "last_updated": current_time,
#         "genesis_block": {
#             "block_number": 0,
#             "timestamp": current_time,
#             "previous_hash": "0" * 64,
#             "data": "Genesis Block - Corruption Reporting System v1.0",
#             "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
#             "nonce": 0
#         },
#         "blocks": [],
#         "total_blocks": 0,
#         "chain_verified": True,
#         "verification_timestamp": current_time,
#         "metadata": {
#             "algorithm": "SHA-256",
#             "encoding": "UTF-8",
#             "block_time_target_seconds": 60,
#             "max_block_size_bytes": 1048576,
#             "description": "Blockchain-inspired hash chain for tamper detection"
#         }
#     }
    
#     # 2. Validators JSON - Validator pool state
#     validators_data = {
#         "version": "1.0.0",
#         "created_at": current_time,
#         "last_updated": current_time,
#         "validators": [],
#         "total_validators": 0,
#         "active_validators": 0,
#         "configuration": {
#             "min_validators": 15,
#             "max_validators": 20,
#             "devils_advocate_ratio": 0.1,
#             "consensus_threshold": 0.66,
#             "validator_weights": {
#                 "default": 1.0,
#                 "expert": 1.5,
#                 "devils_advocate": 1.0
#             },
#             "voting_rounds": 3,
#             "timeout_seconds": 300
#         },
#         "statistics": {
#             "total_votes_cast": 0,
#             "consensus_reached_count": 0,
#             "consensus_failed_count": 0,
#             "average_agreement_ratio": 0.0,
#             "average_convergence_time_seconds": 0.0
#         },
#         "metadata": {
#             "description": "Simulated Byzantine Fault Tolerant validator pool",
#             "consensus_algorithm": "Weighted Majority Voting",
#             "fault_tolerance": "Up to 33% malicious validators"
#         }
#     }
    
#     # 3. Index JSON - Fast lookup index
#     index_data = {
#         "version": "1.0.0",
#         "created_at": current_time,
#         "last_updated": current_time,
#         "submissions": {},
#         "total_submissions": 0,
#         "indices": {
#             "by_status": {
#                 "pending": [],
#                 "processing": [],
#                 "completed": [],
#                 "failed": []
#             },
#             "by_date": {},
#             "by_credibility_score": {
#                 "high": [],
#                 "medium": [],
#                 "low": [],
#                 "unscored": []
#             },
#             "by_pseudonym": {},
#             "coordinated_groups": []
#         },
#         "statistics": {
#             "total_evidence_files": 0,
#             "total_reports_generated": 0,
#             "average_credibility_score": 0.0,
#             "coordination_detected_count": 0,
#             "counter_evidence_submissions": 0
#         },
#         "performance": {
#             "average_processing_time_seconds": 0.0,
#             "cache_hit_ratio": 0.0,
#             "storage_size_bytes": 0,
#             "last_cleanup_timestamp": current_time
#         },
#         "metadata": {
#             "description": "Fast lookup index for submissions and analytics",
#             "index_type": "in-memory-backed",
#             "rebuild_on_startup": True,
#             "auto_optimize": True
#         }
#     }
    
#     # Write files
#     files_to_create = {
#         'chain.json': chain_data,
#         'validators.json': validators_data,
#         'index.json': index_data
#     }
    
#     for filename, data in files_to_create.items():
#         file_path = base_dir / filename
        
#         if not file_path.exists():
#             with open(file_path, 'w', encoding='utf-8') as f:
#                 json.dump(data, f, indent=2, ensure_ascii=False)
#             logger.info(f"✓ Created: {file_path}")
#         else:
#             logger.debug(f"File already exists: {file_path}")



# def main():
#     """Main entry point."""
#     import argparse
    
#     parser = argparse.ArgumentParser(
#         description='Initialize data directory structure'
#     )
#     parser.add_argument(
#         '--data-dir',
#         type=Path,
#         default=Path('backend/data'),
#         help='Base data directory path'
#     )
#     parser.add_argument(
#         '--verify',
#         action='store_true',
#         help='Verify structure without creating'
#     )
    
#     args = parser.parse_args()
    
#     if args.verify:
#         # Verify only
#         is_valid = verify_structure(args.data_dir)
#         exit(0 if is_valid else 1)
#     else:
#         # Create structure
#         create_directory_structure(args.data_dir)
#         add_future_months(args.data_dir)
        
#         # Verify after creation
#         verify_structure(args.data_dir)


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""
Corruption Reporting System - Storage Initialization Script
Version: 1.0.0
Description: Initialize storage directories and data files

This script:
- Creates all required data directories
- Initializes JSON data files (chain.json, validators.json, index.json)
- Sets up directory structure for evidence storage
- Creates placeholder files for Git tracking

Usage:
    python scripts/initialize_storage.py [--reset] [--verbose]
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

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
# DIRECTORY STRUCTURE
# =============================================================================

DIRECTORIES = {
    'backend_data': [
        'backend/data',
        'backend/data/submissions',
        'backend/data/evidence',
        'backend/data/evidence/2026',
        'backend/data/evidence/2026/01',
        'backend/data/evidence/2026/02',
        'backend/data/evidence/2026/03',
        'backend/data/evidence/2026/04',
        'backend/data/evidence/2026/05',
        'backend/data/evidence/2026/06',
        'backend/data/evidence/2026/07',
        'backend/data/evidence/2026/08',
        'backend/data/evidence/2026/09',
        'backend/data/evidence/2026/10',
        'backend/data/evidence/2026/11',
        'backend/data/evidence/2026/12',
        'backend/data/reports',
        'backend/data/cache',
    ],
    'evaluation': [
        'evaluation/datasets',
        'evaluation/datasets/faceforensics',
        'evaluation/datasets/celebdf',
        'evaluation/datasets/synthetic_attacks',
        'evaluation/results',
        'evaluation/results/figures',
    ],
    'tests': [
        'tests/fixtures',
        'tests/fixtures/real_images',
        'tests/fixtures/fake_images',
    ],
    'logs': [
        'logs',
    ]
}

# =============================================================================
# DATA FILE INITIALIZATION
# =============================================================================

def initialize_chain_file(filepath: Path, reset: bool = False) -> bool:
    """
    Initialize hash chain file
    
    Args:
        filepath: Path to chain.json
        reset: Whether to reset existing file
        
    Returns:
        True if successful
    """
    if filepath.exists() and not reset:
        logger.info(f"✓ {filepath} already exists (use --reset to recreate)")
        return True
    
    try:
        # Create genesis block
        genesis_block = {
            'version': '1.0.0',
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'description': 'Hash chain for submission integrity verification',
            'chain': [
                {
                    'block_id': 0,
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'previous_hash': '0' * 64,
                    'data_hash': '0' * 64,
                    'submission_count': 0,
                    'is_genesis': True
                }
            ],
            'metadata': {
                'total_blocks': 1,
                'total_submissions': 0,
                'last_updated': datetime.utcnow().isoformat() + 'Z'
            }
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(genesis_block, f, indent=2)
        
        logger.info(f"✓ Created {filepath} with genesis block")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to create {filepath}: {e}")
        return False

def initialize_validators_file(filepath: Path, reset: bool = False) -> bool:
    """
    Initialize validators file
    
    Args:
        filepath: Path to validators.json
        reset: Whether to reset existing file
        
    Returns:
        True if successful
    """
    if filepath.exists() and not reset:
        logger.info(f"✓ {filepath} already exists (use --reset to recreate)")
        return True
    
    try:
        # Create empty validator pool (will be populated by seed_validators.py)
        validators_data = {
            'version': '1.0.0',
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'description': 'Validator pool for Byzantine consensus simulation',
            'validators': [],
            'metadata': {
                'total_validators': 0,
                'active_validators': 0,
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'consensus_threshold': 0.67,
                'devils_advocate_percentage': 0.10
            }
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(validators_data, f, indent=2)
        
        logger.info(f"✓ Created {filepath} (empty pool - run seed_validators.py to populate)")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to create {filepath}: {e}")
        return False

def initialize_index_file(filepath: Path, reset: bool = False) -> bool:
    """
    Initialize index file
    
    Args:
        filepath: Path to index.json
        reset: Whether to reset existing file
        
    Returns:
        True if successful
    """
    if filepath.exists() and not reset:
        logger.info(f"✓ {filepath} already exists (use --reset to recreate)")
        return True
    
    try:
        # Create empty index
        index_data = {
            'version': '1.0.0',
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'description': 'Fast lookup index for submissions',
            'submissions': {},
            'metadata': {
                'total_submissions': 0,
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'index_size': 0
            }
        }
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(index_data, f, indent=2)
        
        logger.info(f"✓ Created {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to create {filepath}: {e}")
        return False

def create_gitkeep_files():
    """Create .gitkeep files in empty directories"""
    gitkeep_dirs = [
        'backend/data/submissions',
        'backend/data/evidence',
        'backend/data/reports',
        'backend/data/cache',
        'evaluation/datasets/faceforensics',
        'evaluation/datasets/celebdf',
        'evaluation/datasets/synthetic_attacks',
        'evaluation/results/figures',
        'tests/fixtures/real_images',
        'tests/fixtures/fake_images',
    ]
    
    for dir_path in gitkeep_dirs:
        gitkeep_path = PROJECT_ROOT / dir_path / '.gitkeep'
        try:
            gitkeep_path.parent.mkdir(parents=True, exist_ok=True)
            gitkeep_path.touch(exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create .gitkeep in {dir_path}: {e}")

# =============================================================================
# DIRECTORY CREATION
# =============================================================================

def create_directories(verbose: bool = False) -> Dict[str, int]:
    """
    Create all required directories
    
    Args:
        verbose: Whether to log each directory creation
        
    Returns:
        Dictionary with creation statistics
    """
    stats = {
        'created': 0,
        'existed': 0,
        'failed': 0
    }
    
    logger.info("Creating directory structure...")
    
    for category, dirs in DIRECTORIES.items():
        if verbose:
            logger.info(f"\nCategory: {category}")
        
        for dir_path in dirs:
            full_path = PROJECT_ROOT / dir_path
            
            try:
                if full_path.exists():
                    stats['existed'] += 1
                    if verbose:
                        logger.info(f"  ✓ {dir_path} (already exists)")
                else:
                    full_path.mkdir(parents=True, exist_ok=True)
                    stats['created'] += 1
                    if verbose:
                        logger.info(f"  ✓ {dir_path} (created)")
            
            except Exception as e:
                stats['failed'] += 1
                logger.error(f"  ✗ {dir_path} (failed: {e})")
    
    return stats

# =============================================================================
# VERIFICATION
# =============================================================================

def verify_storage() -> bool:
    """
    Verify that storage is properly initialized
    
    Returns:
        True if all checks pass
    """
    logger.info("\nVerifying storage initialization...")
    
    all_ok = True
    
    # Check critical directories
    critical_dirs = [
        'backend/data',
        'backend/data/submissions',
        'backend/data/evidence',
    ]
    
    for dir_path in critical_dirs:
        full_path = PROJECT_ROOT / dir_path
        if full_path.exists() and full_path.is_dir():
            logger.info(f"  ✓ {dir_path}")
        else:
            logger.error(f"  ✗ {dir_path} (missing)")
            all_ok = False
    
    # Check data files
    data_files = [
        'backend/data/chain.json',
        'backend/data/validators.json',
        'backend/data/index.json',
    ]
    
    for file_path in data_files:
        full_path = PROJECT_ROOT / file_path
        if full_path.exists() and full_path.is_file():
            logger.info(f"  ✓ {file_path}")
        else:
            logger.error(f"  ✗ {file_path} (missing)")
            all_ok = False
    
    return all_ok

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Initialize storage for Corruption Reporting System'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset existing data files (WARNING: deletes existing data)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify storage without creating anything'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Print header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Storage Initialization")
    print("=" * 70)
    print("")
    
    # Verify only mode
    if args.verify_only:
        success = verify_storage()
        if success:
            logger.info("\n✓ Storage verification passed")
            return 0
        else:
            logger.error("\n✗ Storage verification failed")
            return 1
    
    # Warn about reset
    if args.reset:
        logger.warning("⚠ WARNING: --reset flag will overwrite existing data files")
        try:
            response = input("Are you sure you want to continue? (yes/NO): ")
            if response.lower() != 'yes':
                logger.info("Reset cancelled by user")
                return 0
        except (KeyboardInterrupt, EOFError):
            logger.info("\nReset cancelled by user")
            return 0
        print("")
    
    # Create directories
    logger.info("Step 1: Creating directories")
    logger.info("-" * 70)
    stats = create_directories(args.verbose)
    
    logger.info(f"\nDirectory creation summary:")
    logger.info(f"  Created: {stats['created']}")
    logger.info(f"  Already existed: {stats['existed']}")
    logger.info(f"  Failed: {stats['failed']}")
    
    if stats['failed'] > 0:
        logger.error("\n✗ Some directories failed to create")
        return 1
    
    # Initialize data files
    logger.info("\nStep 2: Initializing data files")
    logger.info("-" * 70)
    
    data_dir = PROJECT_ROOT / 'backend' / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    results.append(initialize_chain_file(data_dir / 'chain.json', args.reset))
    results.append(initialize_validators_file(data_dir / 'validators.json', args.reset))
    results.append(initialize_index_file(data_dir / 'index.json', args.reset))
    
    # Create .gitkeep files
    logger.info("\nStep 3: Creating .gitkeep files")
    logger.info("-" * 70)
    create_gitkeep_files()
    logger.info("✓ .gitkeep files created")
    
    # Verify storage
    logger.info("\nStep 4: Verification")
    logger.info("-" * 70)
    verification_ok = verify_storage()
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("Initialization Summary")
    logger.info("=" * 70)
    
    if all(results) and verification_ok:
        logger.info("✓ Storage initialized successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run: python scripts/seed_validators.py")
        logger.info("  2. Start backend: cd backend && python -m uvicorn main:app --reload")
        return 0
    else:
        logger.error("✗ Storage initialization failed")
        logger.error("Please check the errors above and try again")
        return 1

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nInitialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
