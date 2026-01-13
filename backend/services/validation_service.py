"""
Validation Service - Input sanitization and validation

Provides:
- File upload validation (type, size, format)
- Text input sanitization
- Evidence type verification
- Malicious content detection
- Input length limits
- Security checks
"""

import logging
import mimetypes
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

# Initialize logger
logger = logging.getLogger(__name__)


class ValidationService:
    """
    Validation Service - Input sanitization and security validation.
    
    Features:
    - File type and size validation
    - Text sanitization (XSS, SQL injection prevention)
    - Evidence format verification
    - Malicious content detection
    - Input length limits
    - Filename sanitization
    """
    
    # File size limits (bytes)
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50 MB
    MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500 MB
    MAX_AUDIO_SIZE = 100 * 1024 * 1024  # 100 MB
    MAX_DOCUMENT_SIZE = 20 * 1024 * 1024  # 20 MB
    
    # Text limits
    MAX_TEXT_LENGTH = 10000  # 10k characters
    MAX_NARRATIVE_LENGTH = 5000  # 5k characters
    
    # Allowed file extensions
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    ALLOWED_AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
    ALLOWED_DOCUMENT_EXTENSIONS = {'.pdf', '.doc', '.docx', '.txt'}
    
    # MIME type mappings
    ALLOWED_MIME_TYPES = {
        'image': {'image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff', 'image/webp'},
        'video': {'video/mp4', 'video/x-msvideo', 'video/quicktime', 'video/x-matroska', 'video/webm'},
        'audio': {'audio/mpeg', 'audio/wav', 'audio/mp4', 'audio/ogg', 'audio/flac'},
        'document': {'application/pdf', 'application/msword', 'text/plain'}
    }
    
    def __init__(self):
        """Initialize validation service."""
        logger.info("ValidationService initialized")
    
    def validate_file_upload(
        self,
        file_path: Path,
        evidence_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate file upload for security and format compliance.
        
        Args:
            file_path: Path to uploaded file
            evidence_type: Type of evidence (image/video/audio/document)
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Check file exists
            if not file_path.exists():
                return False, "File does not exist"
            
            # Check file size
            file_size = file_path.stat().st_size
            
            if evidence_type == 'image':
                if file_size > self.MAX_IMAGE_SIZE:
                    return False, f"Image size exceeds limit ({self.MAX_IMAGE_SIZE // (1024*1024)} MB)"
            elif evidence_type == 'video':
                if file_size > self.MAX_VIDEO_SIZE:
                    return False, f"Video size exceeds limit ({self.MAX_VIDEO_SIZE // (1024*1024)} MB)"
            elif evidence_type == 'audio':
                if file_size > self.MAX_AUDIO_SIZE:
                    return False, f"Audio size exceeds limit ({self.MAX_AUDIO_SIZE // (1024*1024)} MB)"
            elif evidence_type == 'document':
                if file_size > self.MAX_DOCUMENT_SIZE:
                    return False, f"Document size exceeds limit ({self.MAX_DOCUMENT_SIZE // (1024*1024)} MB)"
            else:
                return False, f"Invalid evidence type: {evidence_type}"
            
            # Check file extension
            extension = file_path.suffix.lower()
            
            if evidence_type == 'image' and extension not in self.ALLOWED_IMAGE_EXTENSIONS:
                return False, f"Invalid image extension: {extension}"
            elif evidence_type == 'video' and extension not in self.ALLOWED_VIDEO_EXTENSIONS:
                return False, f"Invalid video extension: {extension}"
            elif evidence_type == 'audio' and extension not in self.ALLOWED_AUDIO_EXTENSIONS:
                return False, f"Invalid audio extension: {extension}"
            elif evidence_type == 'document' and extension not in self.ALLOWED_DOCUMENT_EXTENSIONS:
                return False, f"Invalid document extension: {extension}"
            
            # Verify MIME type
            is_mime_valid, mime_error = self._verify_mime_type(file_path, evidence_type)
            if not is_mime_valid:
                return False, mime_error
            
            # Additional checks for images
            if evidence_type == 'image':
                is_image_valid, image_error = self._verify_image_integrity(file_path)
                if not is_image_valid:
                    return False, image_error
            
            logger.debug(f"File validation passed: {file_path.name} ({evidence_type})")
            return True, None
            
        except Exception as e:
            logger.error(f"File validation failed: {e}")
            return False, f"Validation error: {str(e)}"
    
    def _verify_mime_type(
        self,
        file_path: Path,
        evidence_type: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify file MIME type matches declared type.
        
        Args:
            file_path: Path to file
            evidence_type: Declared evidence type
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            # Guess MIME type from file
            mime_type, _ = mimetypes.guess_type(str(file_path))
            
            if not mime_type:
                # Try reading file header for common types
                mime_type = self._detect_mime_from_header(file_path)
            
            if not mime_type:
                return False, "Could not determine file type"
            
            # Check if MIME type matches evidence type
            allowed_mimes = self.ALLOWED_MIME_TYPES.get(evidence_type, set())
            
            if mime_type not in allowed_mimes:
                return False, f"File type mismatch: expected {evidence_type}, got {mime_type}"
            
            return True, None
            
        except Exception as e:
            logger.warning(f"MIME type verification failed: {e}")
            return True, None  # Don't fail validation on MIME check errors
    
    def _detect_mime_from_header(self, file_path: Path) -> Optional[str]:
        """
        Detect MIME type from file header (magic bytes).
        
        Args:
            file_path: Path to file
            
        Returns:
            str: MIME type or None
        """
        try:
            with open(file_path, 'rb') as f:
                header = f.read(16)
            
            # Common file signatures
            if header.startswith(b'\xff\xd8\xff'):
                return 'image/jpeg'
            elif header.startswith(b'\x89PNG\r\n\x1a\n'):
                return 'image/png'
            elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):
                return 'image/gif'
            elif header.startswith(b'RIFF') and b'WEBP' in header:
                return 'image/webp'
            elif header.startswith(b'%PDF'):
                return 'application/pdf'
            elif header.startswith(b'ID3') or header.startswith(b'\xff\xfb'):
                return 'audio/mpeg'
            
            return None
            
        except Exception as e:
            logger.warning(f"Magic byte detection failed: {e}")
            return None
    
    def _verify_image_integrity(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Verify image can be opened and is not corrupted.
        
        Args:
            file_path: Path to image
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            with Image.open(file_path) as img:
                # Try to load image
                img.verify()
                
                # Check dimensions (reasonable limits)
                if img.size[0] > 10000 or img.size[1] > 10000:
                    return False, "Image dimensions too large (max 10000x10000)"
                
                if img.size[0] < 10 or img.size[1] < 10:
                    return False, "Image dimensions too small (min 10x10)"
            
            return True, None
            
        except Exception as e:
            logger.warning(f"Image integrity check failed: {e}")
            return False, f"Invalid or corrupted image: {str(e)}"
    
    def sanitize_text(self, text: str, max_length: Optional[int] = None) -> str:
        """
        Sanitize text input to prevent XSS and injection attacks.
        
        Args:
            text: Input text
            max_length: Maximum allowed length
            
        Returns:
            str: Sanitized text
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Apply length limit
        if max_length:
            text = text[:max_length]
        else:
            text = text[:self.MAX_TEXT_LENGTH]
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Remove control characters (except newlines and tabs)
        text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def validate_text_narrative(self, narrative: str) -> Tuple[bool, Optional[str]]:
        """
        Validate text narrative for submission.
        
        Args:
            narrative: Text narrative
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not narrative:
            return False, "Narrative cannot be empty"
        
        # Check length
        if len(narrative) < 10:
            return False, "Narrative too short (minimum 10 characters)"
        
        if len(narrative) > self.MAX_NARRATIVE_LENGTH:
            return False, f"Narrative too long (maximum {self.MAX_NARRATIVE_LENGTH} characters)"
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'<script',  # XSS
            r'javascript:',  # XSS
            r'on\w+\s*=',  # Event handlers
            r'eval\s*\(',  # Code execution
            r'exec\s*\(',  # Code execution
        ]
        
        narrative_lower = narrative.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, narrative_lower):
                logger.warning(f"Suspicious pattern detected in narrative: {pattern}")
                return False, "Narrative contains potentially malicious content"
        
        return True, None
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and other attacks.
        
        Args:
            filename: Original filename
            
        Returns:
            str: Sanitized filename
        """
        if not filename:
            return "unnamed_file"
        
        # Get base name (remove any path components)
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        # Allow: alphanumeric, dash, underscore, dot
        safe_chars = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
        
        # Prevent double extensions (.php.jpg)
        safe_chars = safe_chars.replace('..', '_')
        
        # Limit length
        if len(safe_chars) > 255:
            # Preserve extension
            name, ext = safe_chars.rsplit('.', 1) if '.' in safe_chars else (safe_chars, '')
            safe_chars = name[:250] + ('.' + ext if ext else '')
        
        # Ensure not empty
        if not safe_chars or safe_chars == '.':
            safe_chars = "unnamed_file"
        
        return safe_chars
    
    def validate_submission_id(self, submission_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate submission ID format.
        
        Args:
            submission_id: Submission identifier
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not submission_id:
            return False, "Submission ID cannot be empty"
        
        # Check length
        if len(submission_id) < 8 or len(submission_id) > 64:
            return False, "Invalid submission ID length"
        
        # Check format (alphanumeric, dash, underscore only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', submission_id):
            return False, "Submission ID contains invalid characters"
        
        return True, None
    
    def validate_pseudonym(self, pseudonym: str) -> Tuple[bool, Optional[str]]:
        """
        Validate pseudonym format.
        
        Args:
            pseudonym: Pseudonym string
            
        Returns:
            tuple: (is_valid, error_message)
        """
        if not pseudonym:
            return False, "Pseudonym cannot be empty"
        
        # Check length
        if len(pseudonym) < 5 or len(pseudonym) > 100:
            return False, "Invalid pseudonym length"
        
        # Should start with expected prefix
        if not pseudonym.startswith('whistleblower-'):
            return False, "Invalid pseudonym format"
        
        return True, None
    
    def check_sql_injection(self, text: str) -> bool:
        """
        Check text for SQL injection patterns.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if suspicious, False if clean
        """
        if not text:
            return False
        
        text_lower = text.lower()
        
        # Common SQL injection patterns
        sql_patterns = [
            r'union\s+select',
            r';\s*drop\s+table',
            r';\s*delete\s+from',
            r';\s*update\s+',
            r';\s*insert\s+into',
            r'--',  # SQL comment
            r'/\*.*\*/',  # SQL comment
            r'xp_cmdshell',
            r'exec\s*\(',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, text_lower):
                logger.warning(f"SQL injection pattern detected: {pattern}")
                return True
        
        return False
    
    def validate_evidence_type(self, evidence_type: str) -> Tuple[bool, Optional[str]]:
        """
        Validate evidence type.
        
        Args:
            evidence_type: Evidence type string
            
        Returns:
            tuple: (is_valid, error_message)
        """
        valid_types = {'image', 'video', 'audio', 'document'}
        
        if not evidence_type:
            return False, "Evidence type cannot be empty"
        
        if evidence_type.lower() not in valid_types:
            return False, f"Invalid evidence type. Must be one of: {', '.join(valid_types)}"
        
        return True, None
    
    def get_file_size_limit(self, evidence_type: str) -> int:
        """
        Get file size limit for evidence type.
        
        Args:
            evidence_type: Evidence type
            
        Returns:
            int: Size limit in bytes
        """
        limits = {
            'image': self.MAX_IMAGE_SIZE,
            'video': self.MAX_VIDEO_SIZE,
            'audio': self.MAX_AUDIO_SIZE,
            'document': self.MAX_DOCUMENT_SIZE
        }
        
        return limits.get(evidence_type.lower(), self.MAX_DOCUMENT_SIZE)
    
    def get_allowed_extensions(self, evidence_type: str) -> set:
        """
        Get allowed file extensions for evidence type.
        
        Args:
            evidence_type: Evidence type
            
        Returns:
            set: Allowed extensions
        """
        extensions = {
            'image': self.ALLOWED_IMAGE_EXTENSIONS,
            'video': self.ALLOWED_VIDEO_EXTENSIONS,
            'audio': self.ALLOWED_AUDIO_EXTENSIONS,
            'document': self.ALLOWED_DOCUMENT_EXTENSIONS
        }
        
        return extensions.get(evidence_type.lower(), set())
