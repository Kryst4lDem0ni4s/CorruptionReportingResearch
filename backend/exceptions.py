"""
Custom Exceptions - Application-specific exception classes

Provides:
- Base exception class
- Domain-specific exceptions
- HTTP-aware exceptions
- Error codes and messages
"""

from typing import Optional, Dict, Any

from backend.constants import ErrorCode, HTTP_500_INTERNAL_SERVER_ERROR


# ==================== BASE EXCEPTIONS ====================

class BaseApplicationException(Exception):
    """
    Base exception for all application exceptions.
    
    Attributes:
        message: Human-readable error message
        error_code: Machine-readable error code
        status_code: HTTP status code
        details: Additional error details
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        status_code: int = HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize exception.
        
        Args:
            message: Error message
            error_code: Error code
            status_code: HTTP status code
            details: Additional details
        """
        self.message = message
        self.error_code = error_code or ErrorCode.INTERNAL_ERROR
        self.status_code = status_code
        self.details = details or {}
        
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary.
        
        Returns:
            dict: Exception details
        """
        return {
            'error': self.error_code,
            'message': self.message,
            'details': self.details
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"[{self.error_code}] {self.message}"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code='{self.error_code}', "
            f"status_code={self.status_code})"
        )


# ==================== INPUT VALIDATION EXCEPTIONS ====================

class ValidationError(BaseApplicationException):
    """Invalid input data."""
    
    def __init__(
        self,
        message: str = "Invalid input data",
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if field:
            message = f"Invalid value for field '{field}': {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_INPUT,
            status_code=400,
            details=details or {}
        )


class FileValidationError(ValidationError):
    """Invalid file upload."""
    
    def __init__(
        self,
        message: str = "Invalid file",
        file_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if file_type:
            message = f"Invalid {file_type} file: {message}"
        
        super().__init__(message=message, details=details)


class FileTooLargeError(BaseApplicationException):
    """File size exceeds limit."""
    
    def __init__(
        self,
        file_size: int,
        max_size: int,
        file_type: Optional[str] = None
    ):
        message = (
            f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds "
            f"maximum allowed ({max_size / 1024 / 1024:.2f} MB)"
        )
        
        if file_type:
            message = f"{file_type.capitalize()} file too large: {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.FILE_TOO_LARGE,
            status_code=413,
            details={
                'file_size': file_size,
                'max_size': max_size,
                'file_type': file_type
            }
        )


# ==================== AUTHENTICATION/AUTHORIZATION ====================

class UnauthorizedError(BaseApplicationException):
    """Unauthorized access."""
    
    def __init__(
        self,
        message: str = "Unauthorized access",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED,
            status_code=401,
            details=details
        )


class ForbiddenError(BaseApplicationException):
    """Forbidden action."""
    
    def __init__(
        self,
        message: str = "Forbidden action",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.UNAUTHORIZED,
            status_code=403,
            details=details
        )


# ==================== RESOURCE EXCEPTIONS ====================

class NotFoundError(BaseApplicationException):
    """Resource not found."""
    
    def __init__(
        self,
        resource_type: str = "Resource",
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if resource_id:
            message = f"{resource_type} '{resource_id}' not found"
        else:
            message = f"{resource_type} not found"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.NOT_FOUND,
            status_code=404,
            details=details or {}
        )


class SubmissionNotFoundError(NotFoundError):
    """Submission not found."""
    
    def __init__(self, submission_id: str):
        super().__init__(
            resource_type="Submission",
            resource_id=submission_id
        )


class ReportNotFoundError(NotFoundError):
    """Report not found."""
    
    def __init__(self, report_id: str):
        super().__init__(
            resource_type="Report",
            resource_id=report_id
        )


# ==================== RATE LIMITING ====================

class RateLimitError(BaseApplicationException):
    """Rate limit exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if retry_after:
            message = f"{message}. Retry after {retry_after} seconds"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            status_code=429,
            details=details or {'retry_after': retry_after}
        )


# ==================== PROCESSING EXCEPTIONS ====================

class ProcessingError(BaseApplicationException):
    """Error during processing."""
    
    def __init__(
        self,
        message: str = "Processing failed",
        stage: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if stage:
            message = f"Processing failed at {stage}: {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.PROCESSING_FAILED,
            status_code=500,
            details=details or {}
        )


class ModelError(BaseApplicationException):
    """ML model error."""
    
    def __init__(
        self,
        message: str = "Model inference failed",
        model_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if model_name:
            message = f"{model_name} model error: {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.MODEL_ERROR,
            status_code=500,
            details=details or {'model': model_name}
        )


class TimeoutError(BaseApplicationException):
    """Operation timeout."""
    
    def __init__(
        self,
        operation: str = "Operation",
        timeout: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        message = f"{operation} timed out"
        if timeout:
            message = f"{message} after {timeout} seconds"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.TIMEOUT,
            status_code=504,
            details=details or {'timeout': timeout}
        )


# ==================== STORAGE EXCEPTIONS ====================

class StorageError(BaseApplicationException):
    """Storage operation failed."""
    
    def __init__(
        self,
        message: str = "Storage operation failed",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if operation:
            message = f"Storage {operation} failed: {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.STORAGE_ERROR,
            status_code=500,
            details=details or {}
        )


class IntegrityError(BaseApplicationException):
    """Data integrity violation."""
    
    def __init__(
        self,
        message: str = "Data integrity violation",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.INTEGRITY_ERROR,
            status_code=500,
            details=details
        )


class HashChainError(IntegrityError):
    """Hash chain integrity error."""
    
    def __init__(
        self,
        message: str = "Hash chain integrity violation",
        block_number: Optional[int] = None
    ):
        if block_number is not None:
            message = f"{message} at block {block_number}"
        
        super().__init__(
            message=message,
            details={'block_number': block_number}
        )


# ==================== CRYPTOGRAPHY EXCEPTIONS ====================

class CryptoError(BaseApplicationException):
    """Cryptographic operation failed."""
    
    def __init__(
        self,
        message: str = "Cryptographic operation failed",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if operation:
            message = f"Cryptography {operation} failed: {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details or {}
        )


class EncryptionError(CryptoError):
    """Encryption failed."""
    
    def __init__(self, message: str = "Encryption failed"):
        super().__init__(message=message, operation="encryption")


class DecryptionError(CryptoError):
    """Decryption failed."""
    
    def __init__(self, message: str = "Decryption failed"):
        super().__init__(message=message, operation="decryption")


# ==================== CONSENSUS EXCEPTIONS ====================

class ConsensusError(BaseApplicationException):
    """Consensus operation failed."""
    
    def __init__(
        self,
        message: str = "Consensus failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            error_code=ErrorCode.PROCESSING_FAILED,
            status_code=500,
            details=details
        )


class ConsensusNotReachedError(ConsensusError):
    """Consensus not reached."""
    
    def __init__(
        self,
        agreement_ratio: float,
        threshold: float
    ):
        message = (
            f"Consensus not reached: {agreement_ratio:.2%} agreement "
            f"(threshold: {threshold:.2%})"
        )
        
        super().__init__(
            message=message,
            details={
                'agreement_ratio': agreement_ratio,
                'threshold': threshold
            }
        )


# ==================== CONFIGURATION EXCEPTIONS ====================

class ConfigurationError(BaseApplicationException):
    """Configuration error."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        if config_key:
            message = f"Configuration error for '{config_key}': {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=500,
            details=details or {}
        )


# ==================== SERVICE UNAVAILABLE ====================

class ServiceUnavailableError(BaseApplicationException):
    """Service temporarily unavailable."""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        if service:
            message = f"{service} service unavailable: {message}"
        
        super().__init__(
            message=message,
            error_code=ErrorCode.INTERNAL_ERROR,
            status_code=503,
            details={'retry_after': retry_after}
        )
