"""
Hash Chain Service - Blockchain-like immutable chain

Provides:
- SHA-256 hash chain for evidence custody
- Genesis block initialization
- Block addition with chaining
- Chain verification
- Tamper detection
- Proof generation
"""

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from filelock import FileLock

# Initialize logger
logger = logging.getLogger(__name__)


class HashChainService:
    """
    Hash Chain Service - Immutable custody chain.
    
    Features:
    - SHA-256 hash chain (blockchain-like)
    - Genesis block initialization
    - Recursive chaining with previous hash
    - Chain integrity verification
    - Tamper detection
    - Proof of custody generation
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize hash chain service.
        
        Args:
            data_dir: Base data directory (defaults to backend/data)
        """
        if data_dir:
            self.data_dir = Path(data_dir)
        else:
            self.data_dir = Path(__file__).parent.parent / "data"
        
        self.chain_file = self.data_dir / "chain.json"
        self.chain_lock = threading.Lock()
        
        # Initialize chain
        self._initialize_chain()
        
        logger.info(f"HashChainService initialized (chain_file={self.chain_file})")
    
    def _initialize_chain(self) -> None:
        """Initialize hash chain with genesis block if not exists."""
        try:
            if not self.chain_file.exists():
                logger.info("Creating genesis block...")
                
                genesis_block = self._create_genesis_block()
                chain_data = {
                    'blocks': [genesis_block],
                    'version': '1.0.0',
                    'created': datetime.utcnow().isoformat()
                }
                
                self._save_chain(chain_data)
                logger.info(f"Genesis block created: {genesis_block['hash'][:16]}...")
            else:
                # Verify existing chain
                self.verify_chain()
                
        except Exception as e:
            logger.error(f"Failed to initialize chain: {e}")
            raise
    
    def _create_genesis_block(self) -> Dict:
        """
        Create genesis block (first block in chain).
        
        Returns:
            dict: Genesis block
        """
        genesis = {
            'index': 0,
            'timestamp': datetime.utcnow().isoformat(),
            'submission_id': 'GENESIS',
            'evidence_hash': '0' * 64,
            'previous_hash': '0' * 64,
            'nonce': 0
        }
        
        # Calculate genesis hash
        genesis['hash'] = self._calculate_block_hash(genesis)
        
        return genesis
    
    def add_block(
        self,
        submission_id: str,
        evidence_hash: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Add new block to chain.
        
        Args:
            submission_id: Unique submission identifier
            evidence_hash: SHA-256 hash of evidence
            metadata: Optional metadata to include
            
        Returns:
            dict: New block
            
        Raises:
            ValueError: If block addition fails
        """
        try:
            with self.chain_lock:
                chain_data = self.load_chain()
                blocks = chain_data.get('blocks', [])
                
                # Get previous block
                if not blocks:
                    raise ValueError("Chain has no blocks (genesis missing)")
                
                previous_block = blocks[-1]
                
                # Create new block
                new_block = {
                    'index': len(blocks),
                    'timestamp': datetime.utcnow().isoformat(),
                    'submission_id': submission_id,
                    'evidence_hash': evidence_hash,
                    'previous_hash': previous_block['hash'],
                    'metadata': metadata or {}
                }
                
                # Calculate hash
                new_block['hash'] = self._calculate_block_hash(new_block)
                
                # Add to chain
                blocks.append(new_block)
                chain_data['blocks'] = blocks
                
                # Save chain
                self._save_chain(chain_data)
                
                logger.info(
                    f"Added block {new_block['index']} for {submission_id} "
                    f"(hash={new_block['hash'][:16]}...)"
                )
                
                return new_block
                
        except Exception as e:
            logger.error(f"Failed to add block: {e}")
            raise ValueError(f"Failed to add block: {str(e)}")
    
    def verify_chain(self) -> bool:
        """
        Verify integrity of entire chain.
        
        Returns:
            bool: True if chain is valid, False otherwise
            
        Raises:
            ValueError: If chain is corrupted
        """
        try:
            chain_data = self.load_chain()
            blocks = chain_data.get('blocks', [])
            
            if not blocks:
                raise ValueError("Chain is empty")
            
            # Verify genesis block
            genesis = blocks[0]
            if genesis['index'] != 0:
                raise ValueError("Invalid genesis block index")
            
            # Verify each block
            for i in range(1, len(blocks)):
                current_block = blocks[i]
                previous_block = blocks[i - 1]
                
                # Check index continuity
                if current_block['index'] != i:
                    raise ValueError(f"Block {i} has invalid index: {current_block['index']}")
                
                # Check previous hash linkage
                if current_block['previous_hash'] != previous_block['hash']:
                    raise ValueError(
                        f"Block {i} has invalid previous_hash: "
                        f"expected {previous_block['hash']}, "
                        f"got {current_block['previous_hash']}"
                    )
                
                # Verify block hash
                calculated_hash = self._calculate_block_hash(current_block)
                if current_block['hash'] != calculated_hash:
                    raise ValueError(
                        f"Block {i} has invalid hash: "
                        f"expected {calculated_hash}, "
                        f"got {current_block['hash']}"
                    )
            
            logger.debug(f"Chain verified successfully ({len(blocks)} blocks)")
            return True
            
        except ValueError as e:
            logger.error(f"Chain verification failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Chain verification error: {e}")
            return False
    
    def get_proof(self, submission_id: str) -> Dict:
        """
        Get chain proof for submission.
        
        Args:
            submission_id: Submission identifier
            
        Returns:
            dict: Chain proof with block details
        """
        try:
            chain_data = self.load_chain()
            blocks = chain_data.get('blocks', [])
            
            # Find block for submission
            for block in blocks:
                if block.get('submission_id') == submission_id:
                    return {
                        'block_index': block['index'],
                        'block_hash': block['hash'],
                        'previous_hash': block['previous_hash'],
                        'timestamp': block['timestamp'],
                        'evidence_hash': block['evidence_hash'],
                        'chain_length': len(blocks),
                        'verified': True
                    }
            
            logger.debug(f"No proof found for submission {submission_id}")
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get proof: {e}")
            return {}
    
    def get_chain_statistics(self) -> Dict:
        """
        Get chain statistics.
        
        Returns:
            dict: Chain statistics
        """
        try:
            chain_data = self.load_chain()
            blocks = chain_data.get('blocks', [])
            
            # Calculate statistics
            if blocks:
                first_block = blocks[0]
                last_block = blocks[-1]
                
                first_time = datetime.fromisoformat(first_block['timestamp'])
                last_time = datetime.fromisoformat(last_block['timestamp'])
                chain_age_seconds = (last_time - first_time).total_seconds()
            else:
                chain_age_seconds = 0
            
            stats = {
                'total_blocks': len(blocks),
                'chain_version': chain_data.get('version', '1.0.0'),
                'genesis_timestamp': blocks[0]['timestamp'] if blocks else None,
                'latest_timestamp': blocks[-1]['timestamp'] if blocks else None,
                'chain_age_seconds': chain_age_seconds,
                'chain_age_hours': round(chain_age_seconds / 3600, 2),
                'average_block_time_seconds': (
                    chain_age_seconds / (len(blocks) - 1)
                    if len(blocks) > 1 else 0
                ),
                'chain_verified': self.verify_chain(),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get chain statistics: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def load_chain(self) -> Dict:
        """
        Load chain data from file.
        
        Returns:
            dict: Chain data with blocks
        """
        try:
            if not self.chain_file.exists():
                logger.warning("Chain file not found")
                return {'blocks': []}
            
            lock_path = self.chain_file.with_suffix('.lock')
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                with open(self.chain_file, 'r', encoding='utf-8') as f:
                    chain_data = json.load(f)
            
            return chain_data
            
        except Exception as e:
            logger.error(f"Failed to load chain: {e}")
            return {'blocks': []}
    
    def _save_chain(self, chain_data: Dict) -> None:
        """
        Save chain data to file with atomic write.
        
        Args:
            chain_data: Chain data to save
        """
        try:
            lock_path = self.chain_file.with_suffix('.lock')
            lock = FileLock(str(lock_path), timeout=10)
            
            with lock:
                # Write to temp file first
                temp_path = self.chain_file.with_suffix('.tmp')
                
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(chain_data, f, indent=2, ensure_ascii=False)
                
                # Atomic rename
                temp_path.replace(self.chain_file)
                
        except Exception as e:
            logger.error(f"Failed to save chain: {e}")
            raise
    
    def _calculate_block_hash(self, block: Dict) -> str:
        """
        Calculate SHA-256 hash of block.
        
        Args:
            block: Block dictionary
            
        Returns:
            str: Hex-encoded SHA-256 hash
        """
        # Create deterministic string representation
        # Exclude 'hash' field itself
        block_copy = {k: v for k, v in block.items() if k != 'hash'}
        
        # Sort keys for consistency
        block_string = json.dumps(block_copy, sort_keys=True, ensure_ascii=False)
        
        # Calculate SHA-256
        hash_object = hashlib.sha256(block_string.encode('utf-8'))
        return hash_object.hexdigest()
    
    def export_chain(self, output_path: Path) -> bool:
        """
        Export chain to file.
        
        Args:
            output_path: Path to export file
            
        Returns:
            bool: True if successful
        """
        try:
            chain_data = self.load_chain()
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(chain_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Chain exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export chain: {e}")
            return False
    
    def rebuild_chain(self, backup_path: Path) -> bool:
        """
        Rebuild chain from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            bool: True if successful
        """
        try:
            with open(backup_path, 'r', encoding='utf-8') as f:
                chain_data = json.load(f)
            
            # Verify chain before rebuilding
            blocks = chain_data.get('blocks', [])
            
            if not blocks:
                raise ValueError("Backup chain is empty")
            
            # Save backup chain
            self._save_chain(chain_data)
            
            # Verify rebuilt chain
            self.verify_chain()
            
            logger.info(f"Chain rebuilt from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to rebuild chain: {e}")
            return False
