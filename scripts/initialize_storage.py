"""
Initialize Storage - Create data directory structure

Creates all necessary directories and .gitkeep files for the data storage system.
"""

import logging
from pathlib import Path
from datetime import datetime

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure(base_dir: Path = Path('backend/data')):
    """
    Create complete data directory structure.
    
    Args:
        base_dir: Base data directory path
    """
    logger.info(f"Creating data directory structure at: {base_dir}")
    
    # Define directory structure
    directories = [
        # Root data directory
        base_dir,
        
        # Submissions directory
        base_dir / 'submissions',
        
        # Evidence directory with year/month sharding
        base_dir / 'evidence',
        base_dir / 'evidence' / '2026',
        base_dir / 'evidence' / '2026' / '01',
        base_dir / 'evidence' / '2026' / '02',
        base_dir / 'evidence' / '2026' / '03',
        base_dir / 'evidence' / '2026' / '04',
        base_dir / 'evidence' / '2026' / '05',
        base_dir / 'evidence' / '2026' / '06',
        base_dir / 'evidence' / '2026' / '07',
        base_dir / 'evidence' / '2026' / '08',
        base_dir / 'evidence' / '2026' / '09',
        base_dir / 'evidence' / '2026' / '10',
        base_dir / 'evidence' / '2026' / '11',
        base_dir / 'evidence' / '2026' / '12',
        
        # Reports directory
        base_dir / 'reports',
        
        # Cache directory
        base_dir / 'cache',
        
        # Archive directory (for old reports)
        base_dir / 'reports_archive',
    ]
    
    # Create directories
    created_count = 0
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
            created_count += 1
            
            # Create .gitkeep file to track empty directories in Git
            gitkeep_file = directory / '.gitkeep'
            if not gitkeep_file.exists():
                gitkeep_file.touch()
                logger.debug(f"Created .gitkeep: {gitkeep_file}")
        else:
            logger.debug(f"Directory already exists: {directory}")
    
    # Create initial JSON files if they don't exist
    json_files = {
        'chain.json': {
            'genesis_block': {
                'block_number': 0,
                'timestamp': datetime.now().isoformat(),
                'previous_hash': '0' * 64,
                'data': 'Genesis Block',
                'hash': None  # Will be computed on first use
            },
            'blocks': [],
            'created_at': datetime.now().isoformat()
        },
        'validators.json': {
            'validators': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        },
        'index.json': {
            'submissions': {},
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
    }
    
    import json
    
    for filename, initial_data in json_files.items():
        file_path = base_dir / filename
        if not file_path.exists():
            with open(file_path, 'w') as f:
                json.dump(initial_data, f, indent=2)
            logger.info(f"Created initial file: {file_path}")
        else:
            logger.debug(f"File already exists: {file_path}")
    
    logger.info(
        f"Data directory structure initialized successfully! "
        f"Created {created_count} new directories."
    )
    
    return base_dir


def add_future_months(base_dir: Path = Path('backend/data'), year: int = 2026):
    """
    Add month directories for the rest of the year.
    
    Args:
        base_dir: Base data directory
        year: Year for month directories
    """
    evidence_dir = base_dir / 'evidence' / str(year)
    
    for month in range(1, 13):
        month_dir = evidence_dir / f'{month:02d}'
        if not month_dir.exists():
            month_dir.mkdir(parents=True, exist_ok=True)
            (month_dir / '.gitkeep').touch()
            logger.info(f"Created month directory: {month_dir}")


def verify_structure(base_dir: Path = Path('backend/data')) -> bool:
    """
    Verify data directory structure is complete.
    
    Args:
        base_dir: Base data directory
        
    Returns:
        bool: True if structure is valid
    """
    logger.info("Verifying data directory structure...")
    
    required_dirs = [
        base_dir / 'submissions',
        base_dir / 'evidence',
        base_dir / 'reports',
        base_dir / 'cache',
    ]
    
    required_files = [
        base_dir / 'chain.json',
        base_dir / 'validators.json',
        base_dir / 'index.json',
    ]
    
    all_valid = True
    
    # Check directories
    for directory in required_dirs:
        if not directory.exists():
            logger.error(f"Missing directory: {directory}")
            all_valid = False
        else:
            logger.debug(f"✓ Directory exists: {directory}")
    
    # Check files
    for file_path in required_files:
        if not file_path.exists():
            logger.error(f"Missing file: {file_path}")
            all_valid = False
        else:
            logger.debug(f"✓ File exists: {file_path}")
    
    if all_valid:
        logger.info("✓ Data directory structure is valid!")
    else:
        logger.error("✗ Data directory structure has issues!")
    
    return all_valid

def create_initial_json_files(base_dir: Path) -> None:
    """
    Create initial JSON files with proper structure.
    
    Args:
        base_dir: Base data directory
    """
    from datetime import datetime
    import json
    
    logger.info("Creating initial JSON files...")
    
    current_time = datetime.now().isoformat()
    
    # 1. Chain JSON - Hash chain state
    chain_data = {
        "version": "1.0.0",
        "created_at": current_time,
        "last_updated": current_time,
        "genesis_block": {
            "block_number": 0,
            "timestamp": current_time,
            "previous_hash": "0" * 64,
            "data": "Genesis Block - Corruption Reporting System v1.0",
            "hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "nonce": 0
        },
        "blocks": [],
        "total_blocks": 0,
        "chain_verified": True,
        "verification_timestamp": current_time,
        "metadata": {
            "algorithm": "SHA-256",
            "encoding": "UTF-8",
            "block_time_target_seconds": 60,
            "max_block_size_bytes": 1048576,
            "description": "Blockchain-inspired hash chain for tamper detection"
        }
    }
    
    # 2. Validators JSON - Validator pool state
    validators_data = {
        "version": "1.0.0",
        "created_at": current_time,
        "last_updated": current_time,
        "validators": [],
        "total_validators": 0,
        "active_validators": 0,
        "configuration": {
            "min_validators": 15,
            "max_validators": 20,
            "devils_advocate_ratio": 0.1,
            "consensus_threshold": 0.66,
            "validator_weights": {
                "default": 1.0,
                "expert": 1.5,
                "devils_advocate": 1.0
            },
            "voting_rounds": 3,
            "timeout_seconds": 300
        },
        "statistics": {
            "total_votes_cast": 0,
            "consensus_reached_count": 0,
            "consensus_failed_count": 0,
            "average_agreement_ratio": 0.0,
            "average_convergence_time_seconds": 0.0
        },
        "metadata": {
            "description": "Simulated Byzantine Fault Tolerant validator pool",
            "consensus_algorithm": "Weighted Majority Voting",
            "fault_tolerance": "Up to 33% malicious validators"
        }
    }
    
    # 3. Index JSON - Fast lookup index
    index_data = {
        "version": "1.0.0",
        "created_at": current_time,
        "last_updated": current_time,
        "submissions": {},
        "total_submissions": 0,
        "indices": {
            "by_status": {
                "pending": [],
                "processing": [],
                "completed": [],
                "failed": []
            },
            "by_date": {},
            "by_credibility_score": {
                "high": [],
                "medium": [],
                "low": [],
                "unscored": []
            },
            "by_pseudonym": {},
            "coordinated_groups": []
        },
        "statistics": {
            "total_evidence_files": 0,
            "total_reports_generated": 0,
            "average_credibility_score": 0.0,
            "coordination_detected_count": 0,
            "counter_evidence_submissions": 0
        },
        "performance": {
            "average_processing_time_seconds": 0.0,
            "cache_hit_ratio": 0.0,
            "storage_size_bytes": 0,
            "last_cleanup_timestamp": current_time
        },
        "metadata": {
            "description": "Fast lookup index for submissions and analytics",
            "index_type": "in-memory-backed",
            "rebuild_on_startup": True,
            "auto_optimize": True
        }
    }
    
    # Write files
    files_to_create = {
        'chain.json': chain_data,
        'validators.json': validators_data,
        'index.json': index_data
    }
    
    for filename, data in files_to_create.items():
        file_path = base_dir / filename
        
        if not file_path.exists():
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ Created: {file_path}")
        else:
            logger.debug(f"File already exists: {file_path}")



def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Initialize data directory structure'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('backend/data'),
        help='Base data directory path'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify structure without creating'
    )
    
    args = parser.parse_args()
    
    if args.verify:
        # Verify only
        is_valid = verify_structure(args.data_dir)
        exit(0 if is_valid else 1)
    else:
        # Create structure
        create_directory_structure(args.data_dir)
        add_future_months(args.data_dir)
        
        # Verify after creation
        verify_structure(args.data_dir)


if __name__ == '__main__':
    main()
