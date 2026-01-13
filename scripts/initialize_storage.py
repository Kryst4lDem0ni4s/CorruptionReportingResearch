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
