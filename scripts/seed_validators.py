#!/usr/bin/env python3
"""
Corruption Reporting System - Validator Seeding Script
Version: 1.0.0
Description: Generate validator pool for Byzantine consensus simulation

This script creates a pool of simulated validators with:
- Random unique IDs
- Behavioral characteristics (honest, devil's advocate)
- Reliability scores
- Voting weights

Usage:
    python scripts/seed_validators.py [--count N] [--reset]
"""

import os
import sys
import json
import argparse
import logging
import secrets
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

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
# VALIDATOR CONFIGURATION
# =============================================================================

DEFAULT_VALIDATOR_COUNT = 20
MIN_VALIDATORS = 10
MAX_VALIDATORS = 50
DEVILS_ADVOCATE_PERCENTAGE = 0.10  # 10% of validators
CONSENSUS_THRESHOLD = 0.67  # 67% majority required

# Validator types
VALIDATOR_TYPES = {
    'honest': {
        'description': 'Standard validator with high reliability',
        'reliability_range': (0.85, 0.95),
        'weight_range': (0.9, 1.1)
    },
    'devils_advocate': {
        'description': 'Contrarian validator for robustness testing',
        'reliability_range': (0.70, 0.85),
        'weight_range': (0.8, 1.0)
    }
}

# =============================================================================
# VALIDATOR GENERATION
# =============================================================================

def generate_validator_id() -> str:
    """
    Generate unique validator ID
    
    Returns:
        Unique 16-character hex ID
    """
    return secrets.token_hex(8)

def generate_validator(validator_type: str, index: int) -> Dict[str, Any]:
    """
    Generate a single validator
    
    Args:
        validator_type: Type of validator ('honest' or 'devils_advocate')
        index: Validator index for naming
        
    Returns:
        Validator dictionary
    """
    config = VALIDATOR_TYPES[validator_type]
    
    # Generate random values within configured ranges
    reliability = secrets.SystemRandom().uniform(*config['reliability_range'])
    weight = secrets.SystemRandom().uniform(*config['weight_range'])
    
    validator = {
        'validator_id': generate_validator_id(),
        'name': f"Validator_{index:03d}",
        'type': validator_type,
        'reliability_score': round(reliability, 4),
        'voting_weight': round(weight, 4),
        'active': True,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'total_votes': 0,
        'correct_votes': 0,
        'metadata': {
            'description': config['description'],
            'version': '1.0.0'
        }
    }
    
    return validator

def generate_validators(count: int) -> List[Dict[str, Any]]:
    """
    Generate pool of validators
    
    Args:
        count: Number of validators to generate
        
    Returns:
        List of validator dictionaries
    """
    logger.info(f"Generating {count} validators...")
    
    # Calculate devil's advocate count
    devils_advocate_count = max(1, int(count * DEVILS_ADVOCATE_PERCENTAGE))
    honest_count = count - devils_advocate_count
    
    logger.info(f"  Honest validators: {honest_count}")
    logger.info(f"  Devil's advocate validators: {devils_advocate_count}")
    
    validators = []
    
    # Generate honest validators
    for i in range(honest_count):
        validator = generate_validator('honest', i + 1)
        validators.append(validator)
    
    # Generate devil's advocate validators
    for i in range(devils_advocate_count):
        validator = generate_validator('devils_advocate', honest_count + i + 1)
        validators.append(validator)
    
    # Shuffle to randomize order
    secrets.SystemRandom().shuffle(validators)
    
    logger.info(f"✓ Generated {len(validators)} validators")
    
    return validators

# =============================================================================
# STORAGE OPERATIONS
# =============================================================================

def load_validators_file(filepath: Path) -> Dict[str, Any]:
    """
    Load existing validators file
    
    Args:
        filepath: Path to validators.json
        
    Returns:
        Validators data dictionary
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        logger.warning(f"Validators file not found: {filepath}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in validators file: {e}")
        return None

def save_validators_file(filepath: Path, validators: List[Dict[str, Any]]) -> bool:
    """
    Save validators to file
    
    Args:
        filepath: Path to validators.json
        validators: List of validator dictionaries
        
    Returns:
        True if successful
    """
    try:
        # Create full validators data structure
        validators_data = {
            'version': '1.0.0',
            'created_at': datetime.utcnow().isoformat() + 'Z',
            'description': 'Validator pool for Byzantine consensus simulation',
            'validators': validators,
            'metadata': {
                'total_validators': len(validators),
                'active_validators': sum(1 for v in validators if v['active']),
                'honest_validators': sum(1 for v in validators if v['type'] == 'honest'),
                'devils_advocate_validators': sum(1 for v in validators if v['type'] == 'devils_advocate'),
                'last_updated': datetime.utcnow().isoformat() + 'Z',
                'consensus_threshold': CONSENSUS_THRESHOLD,
                'devils_advocate_percentage': DEVILS_ADVOCATE_PERCENTAGE
            }
        }
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file with atomic operation
        temp_file = filepath.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(validators_data, f, indent=2)
        
        # Atomic replace
        temp_file.replace(filepath)
        
        logger.info(f"✓ Saved validators to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"✗ Failed to save validators: {e}")
        return False

# =============================================================================
# VALIDATION
# =============================================================================

def validate_validators(validators: List[Dict[str, Any]]) -> bool:
    """
    Validate generated validators
    
    Args:
        validators: List of validator dictionaries
        
    Returns:
        True if valid
    """
    logger.info("Validating validators...")
    
    all_ok = True
    
    # Check count
    if len(validators) < MIN_VALIDATORS:
        logger.error(f"✗ Too few validators: {len(validators)} < {MIN_VALIDATORS}")
        all_ok = False
    
    if len(validators) > MAX_VALIDATORS:
        logger.error(f"✗ Too many validators: {len(validators)} > {MAX_VALIDATORS}")
        all_ok = False
    
    # Check unique IDs
    validator_ids = [v['validator_id'] for v in validators]
    if len(validator_ids) != len(set(validator_ids)):
        logger.error("✗ Duplicate validator IDs found")
        all_ok = False
    
    # Check types
    valid_types = set(VALIDATOR_TYPES.keys())
    for i, validator in enumerate(validators):
        if validator['type'] not in valid_types:
            logger.error(f"✗ Invalid validator type at index {i}: {validator['type']}")
            all_ok = False
        
        # Check reliability score
        if not (0.0 <= validator['reliability_score'] <= 1.0):
            logger.error(f"✗ Invalid reliability score at index {i}: {validator['reliability_score']}")
            all_ok = False
        
        # Check voting weight
        if validator['voting_weight'] <= 0:
            logger.error(f"✗ Invalid voting weight at index {i}: {validator['voting_weight']}")
            all_ok = False
    
    # Check devil's advocate percentage
    devils_advocate_count = sum(1 for v in validators if v['type'] == 'devils_advocate')
    actual_percentage = devils_advocate_count / len(validators)
    expected_percentage = DEVILS_ADVOCATE_PERCENTAGE
    
    if abs(actual_percentage - expected_percentage) > 0.05:  # 5% tolerance
        logger.warning(f"⚠ Devil's advocate percentage: {actual_percentage:.2%} (expected: {expected_percentage:.2%})")
    
    if all_ok:
        logger.info("✓ Validation passed")
    else:
        logger.error("✗ Validation failed")
    
    return all_ok

# =============================================================================
# STATISTICS
# =============================================================================

def print_statistics(validators: List[Dict[str, Any]]):
    """
    Print validator pool statistics
    
    Args:
        validators: List of validator dictionaries
    """
    logger.info("\n" + "=" * 70)
    logger.info("Validator Pool Statistics")
    logger.info("=" * 70)
    
    # Count by type
    honest_count = sum(1 for v in validators if v['type'] == 'honest')
    devils_advocate_count = sum(1 for v in validators if v['type'] == 'devils_advocate')
    
    logger.info(f"Total Validators: {len(validators)}")
    logger.info(f"  Honest: {honest_count} ({honest_count/len(validators)*100:.1f}%)")
    logger.info(f"  Devil's Advocate: {devils_advocate_count} ({devils_advocate_count/len(validators)*100:.1f}%)")
    
    # Reliability statistics
    reliabilities = [v['reliability_score'] for v in validators]
    avg_reliability = sum(reliabilities) / len(reliabilities)
    min_reliability = min(reliabilities)
    max_reliability = max(reliabilities)
    
    logger.info(f"\nReliability Scores:")
    logger.info(f"  Average: {avg_reliability:.4f}")
    logger.info(f"  Range: [{min_reliability:.4f}, {max_reliability:.4f}]")
    
    # Voting weight statistics
    weights = [v['voting_weight'] for v in validators]
    avg_weight = sum(weights) / len(weights)
    min_weight = min(weights)
    max_weight = max(weights)
    
    logger.info(f"\nVoting Weights:")
    logger.info(f"  Average: {avg_weight:.4f}")
    logger.info(f"  Range: [{min_weight:.4f}, {max_weight:.4f}]")
    
    # Consensus parameters
    logger.info(f"\nConsensus Parameters:")
    logger.info(f"  Threshold: {CONSENSUS_THRESHOLD:.2%} majority")
    logger.info(f"  Minimum votes needed: {int(len(validators) * CONSENSUS_THRESHOLD) + 1}")
    
    logger.info("=" * 70 + "\n")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Generate validator pool for Corruption Reporting System'
    )
    parser.add_argument(
        '--count',
        type=int,
        default=DEFAULT_VALIDATOR_COUNT,
        help=f'Number of validators to generate (default: {DEFAULT_VALIDATOR_COUNT})'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset existing validators (WARNING: deletes existing data)'
    )
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate existing validators without generating'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Custom output file path (default: backend/data/validators.json)'
    )
    
    args = parser.parse_args()
    
    # Validate count
    if args.count < MIN_VALIDATORS or args.count > MAX_VALIDATORS:
        logger.error(f"Validator count must be between {MIN_VALIDATORS} and {MAX_VALIDATORS}")
        return 1
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / 'backend' / 'data' / 'validators.json'
    
    # Print header
    print("\n" + "=" * 70)
    print("Corruption Reporting System - Validator Seeding")
    print("=" * 70)
    print("")
    
    # Validate only mode
    if args.validate_only:
        logger.info("Validation mode - checking existing validators")
        data = load_validators_file(output_path)
        
        if data is None:
            logger.error("No validators file found")
            return 1
        
        validators = data.get('validators', [])
        if not validators:
            logger.error("No validators in file")
            return 1
        
        success = validate_validators(validators)
        if success:
            print_statistics(validators)
            logger.info("✓ Validation successful")
            return 0
        else:
            logger.error("✗ Validation failed")
            return 1
    
    # Check if file exists
    if output_path.exists() and not args.reset:
        logger.info(f"Validators file already exists: {output_path}")
        
        # Load and show existing
        data = load_validators_file(output_path)
        if data:
            existing_count = len(data.get('validators', []))
            logger.info(f"Existing validators: {existing_count}")
        
        logger.info("Use --reset to overwrite existing validators")
        
        try:
            response = input("\nDo you want to add more validators? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                logger.info("Seeding cancelled by user")
                return 0
        except (KeyboardInterrupt, EOFError):
            logger.info("\nSeeding cancelled by user")
            return 0
    
    # Warn about reset
    if args.reset and output_path.exists():
        logger.warning("⚠ WARNING: --reset will delete existing validators")
        try:
            response = input("Are you sure you want to continue? (yes/NO): ")
            if response.lower() != 'yes':
                logger.info("Reset cancelled by user")
                return 0
        except (KeyboardInterrupt, EOFError):
            logger.info("\nReset cancelled by user")
            return 0
        print("")
    
    # Generate validators
    logger.info(f"Generating {args.count} validators...")
    logger.info("")
    
    validators = generate_validators(args.count)
    
    # Validate generated validators
    if not validate_validators(validators):
        logger.error("Generated validators failed validation")
        return 1
    
    # Print statistics
    print_statistics(validators)
    
    # Save to file
    logger.info(f"Saving validators to {output_path}...")
    success = save_validators_file(output_path, validators)
    
    if not success:
        logger.error("Failed to save validators")
        return 1
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("Seeding Complete!")
    logger.info("=" * 70)
    logger.info(f"✓ {len(validators)} validators generated and saved")
    logger.info(f"✓ File: {output_path}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Start backend: cd backend && python -m uvicorn main:app --reload")
    logger.info("  2. The system will use these validators for consensus")
    logger.info("")
    
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("\nSeeding interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
