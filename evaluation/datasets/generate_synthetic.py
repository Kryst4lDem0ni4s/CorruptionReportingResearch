#!/usr/bin/env python3
"""
Corruption Reporting System - Synthetic Attack Generator
Version: 1.0.0
Description: Generate synthetic coordinated attack scenarios for testing

This script generates:
- Coordinated attack groups (3-10 submissions)
- Linguistic similarity patterns
- Temporal clustering patterns
- Style consistency patterns
- Attack-defense pairs

Usage:
    # Generate default attacks
    python generate_synthetic.py
    
    # Generate specific number
    python generate_synthetic.py --num-groups 20
    
    # Specify output
    python generate_synthetic.py --output synthetic_attacks/custom.json
    
    # Include attack-defense pairs
    python generate_synthetic.py --with-defense

Dependencies: None (uses standard library only)
"""

import argparse
import sys
import os
import json
import random
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import dataset package
try:
    from evaluation.datasets import get_dataset_path, logger
except ImportError as e:
    print(f"Error importing datasets package: {e}")
    sys.exit(1)

# ============================================
# CONFIGURATION
# ============================================

# Attack patterns
ATTACK_PATTERNS = [
    'linguistic_similarity',
    'temporal_clustering',
    'style_consistency',
    'coordinated_timing',
    'identical_evidence'
]

# Narrative templates for synthetic submissions
NARRATIVE_TEMPLATES = [
    "Official {name} accepted bribe of ${amount} from {company} on {date}.",
    "{name} misused public funds totaling ${amount} for personal gain.",
    "Corruption scheme involving {name} and {company} discovered.",
    "{name} awarded contract to {company} without proper bidding process.",
    "Evidence shows {name} received illegal payments from {company}.",
    "{name} facilitated illegal deal worth ${amount} with {company}.",
    "Investigation reveals {name} embezzled ${amount} from public funds.",
    "{name} engaged in fraudulent activities with {company}."
]

# Name variations for coordination patterns
OFFICIAL_NAMES = [
    "John Smith", "Jane Doe", "Robert Johnson", "Mary Williams",
    "Michael Brown", "Sarah Davis", "David Miller", "Lisa Wilson"
]

COMPANY_NAMES = [
    "ABC Corporation", "XYZ Industries", "Global Enterprises",
    "National Construction", "Metropolitan Services", "Premier Holdings"
]

# Linguistic patterns for similarity
LINGUISTIC_PATTERNS = {
    'high_similarity': {
        'shared_phrases': [
            "substantial evidence indicates",
            "credible sources confirm",
            "documented proof shows",
            "reliable witnesses attest"
        ],
        'shared_structure': True,
        'similarity_score': 0.85
    },
    'medium_similarity': {
        'shared_phrases': [
            "evidence suggests",
            "reports indicate",
            "sources claim"
        ],
        'shared_structure': False,
        'similarity_score': 0.65
    },
    'low_similarity': {
        'shared_phrases': [],
        'shared_structure': False,
        'similarity_score': 0.35
    }
}

# ============================================
# SYNTHETIC ATTACK GENERATOR
# ============================================

class SyntheticAttackGenerator:
    """Generate synthetic coordinated attack scenarios"""
    
    def __init__(self, seed: int = 42):
        """
        Initialize generator
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        
        # Statistics
        self.stats = {
            'total_groups': 0,
            'total_submissions': 0,
            'patterns': {}
        }
    
    # ========================================
    # MAIN GENERATION
    # ========================================
    
    def generate_attacks(
        self,
        num_groups: int = 10,
        min_group_size: int = 3,
        max_group_size: int = 10,
        with_defense: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate coordinated attack scenarios
        
        Args:
            num_groups: Number of attack groups
            min_group_size: Minimum submissions per group
            max_group_size: Maximum submissions per group
            with_defense: Include defense submissions
            
        Returns:
            List of attack scenario dictionaries
        """
        logger.info(f"Generating {num_groups} attack scenarios...")
        
        attacks = []
        
        for i in range(num_groups):
            # Random group size
            group_size = random.randint(min_group_size, max_group_size)
            
            # Select attack pattern
            pattern = random.choice(ATTACK_PATTERNS)
            
            # Generate attack group
            attack = self._generate_attack_group(
                group_id=i,
                group_size=group_size,
                pattern=pattern
            )
            
            # Add defense if requested
            if with_defense and random.random() > 0.5:
                attack['defense'] = self._generate_defense(attack)
            
            attacks.append(attack)
            
            # Update statistics
            self.stats['total_groups'] += 1
            self.stats['total_submissions'] += group_size
            self.stats['patterns'][pattern] = self.stats['patterns'].get(pattern, 0) + 1
        
        logger.info(f"Generated {len(attacks)} attack scenarios")
        logger.info(f"Total submissions: {self.stats['total_submissions']}")
        
        return attacks
    
    # ========================================
    # ATTACK GROUP GENERATION
    # ========================================
    
    def _generate_attack_group(
        self,
        group_id: int,
        group_size: int,
        pattern: str
    ) -> Dict[str, Any]:
        """Generate single attack group"""
        
        # Base attack information
        target_name = random.choice(OFFICIAL_NAMES)
        company_name = random.choice(COMPANY_NAMES)
        amount = random.randint(10000, 1000000)
        base_date = datetime.now() - timedelta(days=random.randint(30, 365))
        
        # Generate submissions
        submissions = []
        
        for j in range(group_size):
            submission = self._generate_submission(
                submission_id=j,
                pattern=pattern,
                target_name=target_name,
                company_name=company_name,
                amount=amount,
                base_date=base_date,
                group_size=group_size
            )
            submissions.append(submission)
        
        # Calculate coordination metrics
        metrics = self._calculate_coordination_metrics(submissions, pattern)
        
        return {
            'id': f'attack_{group_id:03d}',
            'group_size': group_size,
            'pattern': pattern,
            'target': target_name,
            'submissions': submissions,
            'metrics': metrics,
            'created': datetime.now().isoformat()
        }
    
    def _generate_submission(
        self,
        submission_id: int,
        pattern: str,
        target_name: str,
        company_name: str,
        amount: int,
        base_date: datetime,
        group_size: int
    ) -> Dict[str, Any]:
        """Generate individual submission in attack group"""
        
        # Generate narrative based on pattern
        if pattern == 'linguistic_similarity':
            narrative = self._generate_similar_narrative(
                target_name, company_name, amount, base_date, submission_id
            )
            timestamp = base_date + timedelta(hours=random.randint(0, 72))
            
        elif pattern == 'temporal_clustering':
            narrative = self._generate_varied_narrative(
                target_name, company_name, amount, base_date
            )
            # All within 2-hour window
            timestamp = base_date + timedelta(minutes=random.randint(0, 120))
            
        elif pattern == 'style_consistency':
            narrative = self._generate_styled_narrative(
                target_name, company_name, amount, base_date, style_id=0
            )
            timestamp = base_date + timedelta(hours=random.randint(0, 48))
            
        elif pattern == 'coordinated_timing':
            narrative = self._generate_varied_narrative(
                target_name, company_name, amount, base_date
            )
            # Submissions at same hour on different days
            timestamp = base_date + timedelta(days=submission_id, hours=random.randint(9, 10))
            
        else:  # identical_evidence
            narrative = self._generate_identical_narrative(
                target_name, company_name, amount, base_date
            )
            timestamp = base_date + timedelta(hours=random.randint(0, 168))
        
        # Generate pseudonym
        pseudonym = self._generate_pseudonym(submission_id)
        
        return {
            'id': f'sub_{submission_id:03d}',
            'pseudonym': pseudonym,
            'narrative': narrative,
            'timestamp': timestamp.isoformat(),
            'evidence_type': random.choice(['document', 'image', 'audio', 'video']),
            'metadata': {
                'length': len(narrative),
                'words': len(narrative.split()),
                'pattern_marker': pattern
            }
        }
    
    # ========================================
    # NARRATIVE GENERATION
    # ========================================
    
    def _generate_similar_narrative(
        self,
        name: str,
        company: str,
        amount: int,
        date: datetime,
        idx: int
    ) -> str:
        """Generate narrative with high linguistic similarity"""
        template = random.choice(NARRATIVE_TEMPLATES)
        narrative = template.format(
            name=name,
            company=company,
            amount=f"{amount:,}",
            date=date.strftime("%Y-%m-%d")
        )
        
        # Add shared phrases
        pattern = LINGUISTIC_PATTERNS['high_similarity']
        phrase = random.choice(pattern['shared_phrases'])
        narrative = f"{phrase} that {narrative.lower()}"
        
        # Add slight variation
        if idx > 0:
            variations = [
                f"Additionally, {narrative}",
                f"Furthermore, {narrative}",
                f"Moreover, {narrative}"
            ]
            narrative = random.choice(variations)
        
        return narrative
    
    def _generate_varied_narrative(
        self,
        name: str,
        company: str,
        amount: int,
        date: datetime
    ) -> str:
        """Generate narrative with variation"""
        template = random.choice(NARRATIVE_TEMPLATES)
        narrative = template.format(
            name=name,
            company=company,
            amount=f"{amount:,}",
            date=date.strftime("%Y-%m-%d")
        )
        return narrative
    
    def _generate_styled_narrative(
        self,
        name: str,
        company: str,
        amount: int,
        date: datetime,
        style_id: int
    ) -> str:
        """Generate narrative with consistent style"""
        template = random.choice(NARRATIVE_TEMPLATES)
        narrative = template.format(
            name=name,
            company=company,
            amount=f"{amount:,}",
            date=date.strftime("%Y-%m-%d")
        )
        
        # Apply consistent style markers
        style_markers = {
            0: lambda s: s.upper(),  # All caps
            1: lambda s: f"URGENT: {s}",  # Urgency marker
            2: lambda s: f"{s} [VERIFIED]"  # Verification claim
        }
        
        style_func = style_markers.get(style_id % len(style_markers))
        return style_func(narrative)
    
    def _generate_identical_narrative(
        self,
        name: str,
        company: str,
        amount: int,
        date: datetime
    ) -> str:
        """Generate identical narrative (exact copies)"""
        # Use same template and values
        template = NARRATIVE_TEMPLATES[0]
        return template.format(
            name=name,
            company=company,
            amount=f"{amount:,}",
            date=date.strftime("%Y-%m-%d")
        )
    
    # ========================================
    # DEFENSE GENERATION
    # ========================================
    
    def _generate_defense(self, attack: Dict[str, Any]) -> Dict[str, Any]:
        """Generate counter-evidence/defense submission"""
        target = attack['target']
        
        defense_narratives = [
            f"{target} denies all allegations and provides alibi documentation.",
            f"Character witnesses confirm {target}'s integrity and ethical conduct.",
            f"Financial records contradict corruption claims against {target}.",
            f"{target} presents evidence of lawful transaction procedures.",
            f"Independent audit clears {target} of all wrongdoing."
        ]
        
        return {
            'id': f"defense_{attack['id']}",
            'narrative': random.choice(defense_narratives),
            'timestamp': (datetime.now() + timedelta(hours=random.randint(1, 24))).isoformat(),
            'evidence_type': random.choice(['document', 'witness_statement']),
            'verified_identity': random.random() > 0.5,  # 50% verified
            'credibility_score': random.uniform(0.6, 0.95)
        }
    
    # ========================================
    # METRICS CALCULATION
    # ========================================
    
    def _calculate_coordination_metrics(
        self,
        submissions: List[Dict],
        pattern: str
    ) -> Dict[str, Any]:
        """Calculate coordination detection metrics"""
        
        # Temporal metrics
        timestamps = [datetime.fromisoformat(s['timestamp']) for s in submissions]
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600 
                     for i in range(len(timestamps)-1)]
        avg_time_diff = sum(time_diffs) / len(time_diffs) if time_diffs else 0
        
        # Linguistic metrics
        narratives = [s['narrative'] for s in submissions]
        avg_length = sum(len(n) for n in narratives) / len(narratives)
        
        # Similarity score (pattern-dependent)
        if pattern == 'linguistic_similarity':
            similarity_score = random.uniform(0.80, 0.95)
        elif pattern == 'identical_evidence':
            similarity_score = 1.0
        elif pattern == 'temporal_clustering':
            similarity_score = random.uniform(0.60, 0.75)
        else:
            similarity_score = random.uniform(0.40, 0.70)
        
        return {
            'pattern': pattern,
            'similarity_score': round(similarity_score, 3),
            'temporal_window_hours': round(max(time_diffs) if time_diffs else 0, 2),
            'avg_time_diff_hours': round(avg_time_diff, 2),
            'avg_narrative_length': round(avg_length, 1),
            'detection_difficulty': 'high' if similarity_score > 0.8 else 'medium' if similarity_score > 0.6 else 'low'
        }
    
    # ========================================
    # UTILITIES
    # ========================================
    
    def _generate_pseudonym(self, idx: int) -> str:
        """Generate pseudonym hash"""
        data = f"synthetic_user_{idx}_{self.seed}".encode()
        return hashlib.sha256(data).hexdigest()[:16]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return self.stats.copy()

# ============================================
# COMMAND LINE INTERFACE
# ============================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic coordinated attack scenarios',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default attacks (10 groups)
  python generate_synthetic.py
  
  # Generate 20 attack groups
  python generate_synthetic.py --num-groups 20
  
  # Include defense submissions
  python generate_synthetic.py --with-defense
  
  # Custom output file
  python generate_synthetic.py --output custom_attacks.json
  
  # Vary group sizes
  python generate_synthetic.py --min-size 5 --max-size 15
        """
    )
    
    parser.add_argument(
        '--num-groups',
        type=int,
        default=10,
        help='Number of attack groups to generate (default: 10)'
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        default=3,
        help='Minimum group size (default: 3)'
    )
    
    parser.add_argument(
        '--max-size',
        type=int,
        default=10,
        help='Maximum group size (default: 10)'
    )
    
    parser.add_argument(
        '--with-defense',
        action='store_true',
        help='Include defense submissions'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: synthetic_attacks/synthetic_attacks.json)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    """Main generation function"""
    args = parse_args()
    
    # Setup logging
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create generator
    generator = SyntheticAttackGenerator(seed=args.seed)
    
    # Generate attacks
    attacks = generator.generate_attacks(
        num_groups=args.num_groups,
        min_group_size=args.min_size,
        max_group_size=args.max_size,
        with_defense=args.with_defense
    )
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        dataset_path = get_dataset_path('synthetic_attacks')
        output_path = dataset_path / 'synthetic_attacks.json'
    
    # Create parent directory
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save attacks
    with open(output_path, 'w') as f:
        json.dump(attacks, f, indent=2)
    
    logger.info(f"Saved {len(attacks)} attack scenarios to: {output_path}")
    
    # Print statistics
    stats = generator.get_statistics()
    logger.info("\nGeneration Statistics:")
    logger.info(f"  Total groups: {stats['total_groups']}")
    logger.info(f"  Total submissions: {stats['total_submissions']}")
    logger.info(f"  Pattern distribution:")
    for pattern, count in stats['patterns'].items():
        logger.info(f"    {pattern}: {count}")
    
    # Create README
    readme_path = output_path.parent / 'README.md'
    readme_content = f"""# Synthetic Coordinated Attacks

## Description

Generated synthetic attack scenarios for testing coordination detection.

## Statistics

- Attack groups: {stats['total_groups']}
- Total submissions: {stats['total_submissions']}
- Group sizes: {args.min_size}-{args.max_size}
- Patterns: {', '.join(ATTACK_PATTERNS)}

## Pattern Distribution

"""
    
    for pattern, count in stats['patterns'].items():
        readme_content += f"- {pattern}: {count}\n"
    
    readme_content += f"""
## Data Format

```json
{{
  "id": "attack_001",
  "group_size": 5,
  "pattern": "linguistic_similarity",
  "target": "John Smith",
  "submissions": [...],
  "metrics": {{
    "similarity_score": 0.85,
    "temporal_window_hours": 2.5
  }}
}}

# Command-line usage
# python evaluation/datasets/generate_synthetic.py --num-groups 20 --with-defense