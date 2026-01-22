"""
Layer 1: Anonymous Submission Gateway
Handles pseudonym generation, encryption, hash chains, and metadata stripping.

Input: Raw evidence file + metadata
Output: Anonymized submission with pseudonym and encrypted evidence
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

from PIL import Image
from backend.services.metrics_service import MetricsService

# Initialize logger
logger = logging.getLogger(__name__)


class Layer1Anonymity:
    """
    Layer 1: Anonymous Submission Gateway
    
    Implements:
    - SHA-256 pseudonym generation
    - AES-256 file encryption
    - Hash chain entry creation
    - EXIF metadata stripping
    - Submission data anonymization
    """
    
    def __init__(
        self,
        crypto_service,
        hash_chain_service,
        metadata_service,
        storage_service,
        metrics_service: Optional[MetricsService] = None
    ):
        """
        Initialize Layer 1 with required services.
        
        Args:
            crypto_service: Cryptography service for encryption
            hash_chain_service: Hash chain management service
            metadata_service: Metadata stripping service
            storage_service: Storage service for file operations
        """
        self.crypto = crypto_service
        self.hash_chain = hash_chain_service
        self.metadata = metadata_service
        self.storage = storage_service
        self.metrics = metrics_service
        
        logger.info("Layer 1 (Anonymity) initialized")
    
    def process(
        self,
        submission_id: str,
        file_path: Path,
        evidence_type: str,
        text_narrative: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Process submission through anonymity layer.
        
        Args:
            submission_id: Unique submission identifier
            file_path: Path to evidence file
            evidence_type: Type of evidence (image/audio/video)
            text_narrative: Optional text narrative
            metadata: Optional metadata dictionary
            
        Returns:
            dict: Anonymized submission data
            
        Raises:
            ValueError: If processing fails
        """
        logger.info(f"Layer 1 processing submission {submission_id}")
        
        try:
            # Step 1: Generate pseudonym
            pseudonym = self._generate_pseudonym(submission_id)
            logger.debug(f"Generated pseudonym: {pseudonym}")
            
            # Step 2: Strip metadata from evidence file
            cleaned_file_path = self._strip_metadata(
                file_path, 
                evidence_type
            )
            logger.debug(f"Metadata stripped from evidence")
            
            if self.metrics:
                self.metrics.record_anonymity_violation()
            
            # Step 3: Compute evidence hash (before encryption)
            evidence_hash = self._compute_evidence_hash(cleaned_file_path)
            logger.debug(f"Evidence hash: {evidence_hash[:16]}...")
            
            # Step 4: Encrypt evidence file
            encrypted_file_path = self._encrypt_evidence(
                cleaned_file_path,
                submission_id
            )
            logger.debug(f"Evidence encrypted")
            
            # Step 5: Add entry to hash chain
            # Use correct method add_block which generates its own timestamp
            block = self.hash_chain.add_block(
                submission_id=submission_id,
                evidence_hash=evidence_hash
            )
            chain_hash = block['hash']
            logger.debug(f"Chain hash: {chain_hash[:16]}...")
            
            # Step 6: Sanitize text narrative
            sanitized_narrative = self._sanitize_text(text_narrative)
            
            # Step 7: Anonymize metadata
            anonymized_metadata = self._anonymize_metadata(metadata)
            
            # Step 8: Build anonymized submission data
            anonymized_data = {
                "id": submission_id,
                "pseudonym": pseudonym,
                "evidence_hash": evidence_hash,
                "chain_hash": chain_hash,
                "evidence_type": evidence_type,
                "text_narrative": sanitized_narrative,
                "metadata": anonymized_metadata,
                "encrypted_file_path": str(encrypted_file_path),
                "original_file_path": str(file_path),
                "timestamp_anonymized": datetime.utcnow().isoformat(),
                "anonymity_version": "1.0",
                "layer1_status": "completed"
            }
            
            logger.info(
                f"Layer 1 completed for {submission_id} "
                f"(pseudonym: {pseudonym})"
            )
            
            return anonymized_data
            
        except Exception as e:
            logger.error(f"Layer 1 processing failed for {submission_id}: {e}", exc_info=True)

            if self.metrics:
                self.metrics.record_anonymity_violation()

            raise ValueError(f"Anonymity processing failed: {str(e)}")
        
    def _generate_pseudonym(self, submission_id: str) -> str:
        """
        Generate anonymous pseudonym from submission ID.
        
        Uses SHA-256 hash truncated to 16 characters for anonymity.
        
        Args:
            submission_id: Unique submission identifier
            
        Returns:
            str: 16-character pseudonym
        """
        # Generate salt from submission ID
        salt = submission_id.encode('utf-8')
        
        # Use crypto service to generate pseudonym
        pseudonym = self.crypto.generate_pseudonym(salt)
        
        return pseudonym
    
    def _strip_metadata(
        self, 
        file_path: Path, 
        evidence_type: str
    ) -> Path:
        """
        Strip EXIF and other metadata from evidence file.
        
        Args:
            file_path: Path to original file
            evidence_type: Type of evidence
            
        Returns:
            Path: Path to cleaned file
        """
        try:
            if evidence_type == "image":
                return self.metadata.strip_image_metadata(file_path)
            elif evidence_type == "audio":
                return self.metadata.strip_audio_metadata(file_path)
            elif evidence_type == "video":
                return self.metadata.strip_video_metadata(file_path)
            else:
                # For text or unknown types, return original
                logger.warning(f"Unknown evidence type: {evidence_type}")
                return file_path
                
        except Exception as e:
            logger.warning(
                f"Metadata stripping failed, using original: {e}"
            )
            # Fail gracefully - use original file
            return file_path
    
    def _compute_evidence_hash(self, file_path: Path) -> str:
        """
        Compute SHA-256 hash of evidence file.
        
        Args:
            file_path: Path to evidence file
            
        Returns:
            str: SHA-256 hash (hex)
        """
        return self.crypto.hash_file(file_path)
    
    def _encrypt_evidence(
        self, 
        file_path: Path, 
        submission_id: str
    ) -> Path:
        """
        Encrypt evidence file using AES-256.
        
        Args:
            file_path: Path to file to encrypt
            submission_id: Submission identifier
            
        Returns:
            Path: Path to encrypted file
        """
        # Read file content
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # Encrypt content
        encrypted_content = self.crypto.encrypt_data(content)
        
        # Save encrypted file
        encrypted_path = file_path.parent / f"{submission_id}_encrypted.bin"
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_content)
        
        return encrypted_path
    
    def _sanitize_text(self, text: Optional[str]) -> Optional[str]:
        """
        Sanitize text narrative to remove identifying information.
        
        Args:
            text: Raw text narrative
            
        Returns:
            str: Sanitized text
        """
        if not text:
            return None
        
        # Remove HTML tags
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Optional: Redact common PII patterns (phone, email, etc.)
        # Phone numbers
        text = re.sub(
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            '[PHONE_REDACTED]',
            text
        )
        
        # Email addresses
        text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            '[EMAIL_REDACTED]',
            text
        )
        
        # Limit length
        max_length = 5000
        if len(text) > max_length:
            text = text[:max_length] + "... [TRUNCATED]"
        
        return text
    
    def _anonymize_metadata(self, metadata: Optional[Dict]) -> Dict:
        """
        Anonymize metadata by removing identifying fields.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            dict: Anonymized metadata
        """
        if not metadata:
            return {}
        
        # Allowed metadata fields (whitelist approach)
        allowed_fields = {
            'incident_date',
            'location_city',
            'location_country',
            'category',
            'severity',
            'evidence_count'
        }
        
        # Filter metadata to only allowed fields
        anonymized = {
            k: v for k, v in metadata.items()
            if k in allowed_fields
        }
        
        # Redact precise locations (keep only city/country)
        if 'location_address' in metadata:
            anonymized['location_note'] = 'Address provided but redacted'
        
        return anonymized
    
    def decrypt_evidence(
        self, 
        encrypted_file_path: Path, 
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Decrypt evidence file for processing.
        
        Args:
            encrypted_file_path: Path to encrypted file
            output_path: Optional output path for decrypted file
            
        Returns:
            Path: Path to decrypted file
        """
        # Read encrypted content
        with open(encrypted_file_path, 'rb') as f:
            encrypted_content = f.read()
        
        # Decrypt content
        decrypted_content = self.crypto.decrypt_data(encrypted_content)
        
        # Determine output path
        if not output_path:
            output_path = encrypted_file_path.parent / \
                         f"{encrypted_file_path.stem}_decrypted"
        
        # Save decrypted file
        with open(output_path, 'wb') as f:
            f.write(decrypted_content)
        
        logger.debug(f"Evidence decrypted to {output_path}")
        
        return output_path
    
    def verify_hash_chain(self, submission_id: str, evidence_hash: str) -> bool:
        """
        Verify submission exists in hash chain with correct hash.
        
        Args:
            submission_id: Submission identifier
            evidence_hash: Expected evidence hash
            
        Returns:
            bool: True if verification succeeds
        """
        try:
            return self.hash_chain.verify_entry(submission_id, evidence_hash)
        except Exception as e:
            logger.error(f"Hash chain verification failed: {e}")
            return False
    
    def get_chain_proof(self, submission_id: str) -> Dict:
        """
        Get hash chain proof for submission.
        
        Args:
            submission_id: Submission identifier
            
        Returns:
            dict: Chain proof data
        """
        try:
            return self.hash_chain.get_proof(submission_id)
        except Exception as e:
            logger.error(f"Failed to get chain proof: {e}")
            return {}
