"""
Crypto Service - Cryptographic operations

Provides:
- AES-256-CBC encryption via Fernet
- SHA-256 hashing for pseudonyms
- Secure key generation and management
- Evidence hash calculation
- Encryption/decryption for sensitive data
"""

import hashlib
import logging
import secrets
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Initialize logger
logger = logging.getLogger(__name__)


class CryptoService:
    """
    Crypto Service - Cryptographic operations.
    
    Features:
    - AES-256-CBC encryption using Fernet
    - SHA-256 hashing for pseudonyms and evidence
    - Secure random generation
    - Key derivation
    - Evidence integrity verification
    """
    
    def __init__(self, master_key: Optional[bytes] = None):
        """
        Initialize crypto service.
        
        Args:
            master_key: Master encryption key (32 bytes).
                       If None, generates a new key (for testing only).
        """
        if master_key:
            self.master_key = master_key
        else:
            # Generate new key (WARNING: for testing/development only)
            self.master_key = Fernet.generate_key()
            logger.warning(
                "CryptoService initialized with auto-generated key. "
                "This should only be used for testing!"
            )
        
        # Initialize Fernet cipher
        self.cipher = Fernet(self.master_key)
        
        logger.info("CryptoService initialized")
    
    def generate_pseudonym(self, submission_id: str, salt: Optional[str] = None) -> str:
        """
        Generate anonymous pseudonym using SHA-256.
        
        Creates a collision-resistant pseudonym that preserves anonymity
        while allowing submission tracking.
        
        Args:
            submission_id: Unique submission identifier
            salt: Optional salt for additional randomness
            
        Returns:
            str: Hex-encoded pseudonym (e.g., "whistleblower-a3b5c7...")
        """
        try:
            # Create input for hash
            hash_input = submission_id.encode('utf-8')
            
            # Add salt if provided
            if salt:
                hash_input += salt.encode('utf-8')
            else:
                # Generate deterministic salt from submission_id
                salt_hash = hashlib.sha256(submission_id.encode()).digest()
                hash_input += salt_hash
            
            # Calculate SHA-256 hash
            hash_object = hashlib.sha256(hash_input)
            hash_hex = hash_object.hexdigest()
            
            # Create readable pseudonym (first 12 chars for readability)
            pseudonym = f"whistleblower-{hash_hex[:12]}"
            
            logger.debug(f"Generated pseudonym: {pseudonym}")
            
            return pseudonym
            
        except Exception as e:
            logger.error(f"Failed to generate pseudonym: {e}")
            # Fallback pseudonym
            return f"whistleblower-{secrets.token_hex(6)}"
    
    def hash_data(self, data: bytes) -> str:
        """
        Calculate SHA-256 hash of data.
        
        Args:
            data: Raw data bytes
            
        Returns:
            str: Hex-encoded SHA-256 hash
        """
        try:
            hash_object = hashlib.sha256(data)
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.error(f"Failed to hash data: {e}")
            raise ValueError(f"Hash calculation failed: {str(e)}")
    
    def hash_file(self, file_path: Path) -> str:
        """
        Calculate SHA-256 hash of file.
        
        Uses streaming to handle large files efficiently.
        
        Args:
            file_path: Path to file
            
        Returns:
            str: Hex-encoded SHA-256 hash
            
        Raises:
            FileNotFoundError: If file doesn't exist
            IOError: If file read fails
        """
        try:
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            hash_object = hashlib.sha256()
            
            # Stream file in chunks (64KB)
            chunk_size = 65536
            
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    hash_object.update(chunk)
            
            file_hash = hash_object.hexdigest()
            
            logger.debug(f"File hash: {file_hash[:16]}... (size={file_path.stat().st_size} bytes)")
            
            return file_hash
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            raise IOError(f"File hash failed: {str(e)}")
    
    def encrypt_data(self, data: bytes) -> bytes:
        """
        Encrypt data using AES-256-CBC (via Fernet).
        
        Args:
            data: Raw data to encrypt
            
        Returns:
            bytes: Encrypted data (includes IV and MAC)
            
        Raises:
            ValueError: If encryption fails
        """
        try:
            encrypted = self.cipher.encrypt(data)
            logger.debug(f"Encrypted {len(data)} bytes → {len(encrypted)} bytes")
            return encrypted
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ValueError(f"Encryption failed: {str(e)}")
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """
        Decrypt data encrypted with encrypt_data().
        
        Args:
            encrypted_data: Encrypted data
            
        Returns:
            bytes: Decrypted data
            
        Raises:
            ValueError: If decryption fails (wrong key or corrupted data)
        """
        try:
            decrypted = self.cipher.decrypt(encrypted_data)
            logger.debug(f"Decrypted {len(encrypted_data)} bytes → {len(decrypted)} bytes")
            return decrypted
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ValueError(f"Decryption failed (invalid key or corrupted data): {str(e)}")
    
    def encrypt_file(self, input_path: Path, output_path: Path) -> str:
        """
        Encrypt file and save to output path.
        
        Args:
            input_path: Path to file to encrypt
            output_path: Path to save encrypted file
            
        Returns:
            str: SHA-256 hash of original file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            IOError: If encryption or write fails
        """
        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Read file
            with open(input_path, 'rb') as f:
                data = f.read()
            
            # Calculate hash before encryption
            file_hash = self.hash_data(data)
            
            # Encrypt
            encrypted = self.encrypt_data(data)
            
            # Write encrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(encrypted)
            
            logger.info(
                f"File encrypted: {input_path.name} → {output_path.name} "
                f"({len(data)} → {len(encrypted)} bytes)"
            )
            
            return file_hash
            
        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(f"File encryption failed: {e}")
            raise IOError(f"File encryption failed: {str(e)}")
    
    def decrypt_file(self, input_path: Path, output_path: Path) -> None:
        """
        Decrypt file and save to output path.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to save decrypted file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If decryption fails
            IOError: If write fails
        """
        try:
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Read encrypted file
            with open(input_path, 'rb') as f:
                encrypted = f.read()
            
            # Decrypt
            decrypted = self.decrypt_data(encrypted)
            
            # Write decrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(decrypted)
            
            logger.info(
                f"File decrypted: {input_path.name} → {output_path.name} "
                f"({len(encrypted)} → {len(decrypted)} bytes)"
            )
            
        except (FileNotFoundError, ValueError):
            raise
        except Exception as e:
            logger.error(f"File decryption failed: {e}")
            raise IOError(f"File decryption failed: {str(e)}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate cryptographically secure random token.
        
        Args:
            length: Token length in bytes (default: 32)
            
        Returns:
            str: Hex-encoded random token
        """
        return secrets.token_hex(length)
    
    def derive_key(self, password: str, salt: bytes, iterations: int = 100000) -> bytes:
        """
        Derive encryption key from password using PBKDF2.
        
        Args:
            password: User password
            salt: Random salt (at least 16 bytes)
            iterations: PBKDF2 iteration count
            
        Returns:
            bytes: Derived key (32 bytes)
        """
        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations
            )
            
            key = kdf.derive(password.encode('utf-8'))
            
            logger.debug(f"Key derived from password (iterations={iterations})")
            
            return key
            
        except Exception as e:
            logger.error(f"Key derivation failed: {e}")
            raise ValueError(f"Key derivation failed: {str(e)}")
    
    def verify_hash(self, data: bytes, expected_hash: str) -> bool:
        """
        Verify data matches expected hash.
        
        Args:
            data: Data to verify
            expected_hash: Expected SHA-256 hash (hex)
            
        Returns:
            bool: True if hash matches, False otherwise
        """
        try:
            actual_hash = self.hash_data(data)
            return secrets.compare_digest(actual_hash, expected_hash)
            
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    def create_submission_hash_chain(self, *data_items: bytes) -> str:
        """
        Create chained hash of multiple data items.
        
        Useful for creating composite hashes of submission + evidence + metadata.
        
        Args:
            *data_items: Variable number of data items to hash
            
        Returns:
            str: Hex-encoded chained hash
        """
        try:
            hash_object = hashlib.sha256()
            
            for item in data_items:
                hash_object.update(item)
            
            return hash_object.hexdigest()
            
        except Exception as e:
            logger.error(f"Chain hash failed: {e}")
            raise ValueError(f"Chain hash failed: {str(e)}")
    
    @staticmethod
    def generate_master_key() -> bytes:
        """
        Generate new master encryption key.
        
        WARNING: This key must be securely stored! Loss of key means
        loss of all encrypted data.
        
        Returns:
            bytes: 32-byte master key suitable for Fernet
        """
        return Fernet.generate_key()
    
    @staticmethod
    def key_to_string(key: bytes) -> str:
        """
        Convert key to base64 string for storage.
        
        Args:
            key: Encryption key
            
        Returns:
            str: Base64-encoded key
        """
        return key.decode('utf-8')
    
    @staticmethod
    def string_to_key(key_string: str) -> bytes:
        """
        Convert base64 string back to key.
        
        Args:
            key_string: Base64-encoded key
            
        Returns:
            bytes: Encryption key
        """
        return key_string.encode('utf-8')
