"""
Global Constants - System-wide configuration values

Defines all constant values used across the application:
- Model configurations
- API limits
- Cryptographic parameters
- File size limits
- Timeouts
- Status codes
"""

from enum import Enum
from pathlib import Path


# ==================== SYSTEM METADATA ====================

APP_NAME = "Corruption Reporting System"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Anonymous evidence validation with AI-powered credibility assessment"
API_VERSION = "v1"


# ==================== PATHS ====================

# Base directories
BASE_DIR = Path(__file__).parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BACKEND_DIR / "data"
FRONTEND_DIR = BASE_DIR / "frontend"

# Data subdirectories
SUBMISSIONS_DIR = DATA_DIR / "submissions"
EVIDENCE_DIR = DATA_DIR / "evidence"
REPORTS_DIR = DATA_DIR / "reports"
CACHE_DIR = DATA_DIR / "cache"

# Storage files
CHAIN_FILE = DATA_DIR / "chain.json"
VALIDATORS_FILE = DATA_DIR / "validators.json"
INDEX_FILE = DATA_DIR / "index.json"


# ==================== ML MODELS ====================

# Model identifiers
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
WAV2VEC_MODEL_NAME = "facebook/wav2vec2-base"
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
SENTENCE_TRANSFORMER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model sizes (approximate, in MB)
MODEL_SIZES = {
    "clip": 350,
    "wav2vec": 360,
    "blip": 500,
    "sentence_transformer": 80
}

# Model embedding dimensions
CLIP_EMBEDDING_DIM = 512
WAV2VEC_EMBEDDING_DIM = 768
SENTENCE_TRANSFORMER_EMBEDDING_DIM = 384

# Model inference settings
DEFAULT_BATCH_SIZE = 8
MAX_BATCH_SIZE = 32
USE_FP16 = True  # Use FP16 precision for faster inference
MODEL_CACHE_SIZE = 1000  # Max cached embeddings


# ==================== FILE UPLOAD LIMITS ====================

# File size limits (in bytes)
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_AUDIO_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50 MB
MAX_DOCUMENT_SIZE = 5 * 1024 * 1024  # 5 MB
MAX_TOTAL_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB per submission

# Allowed file extensions
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}

# MIME types
ALLOWED_IMAGE_MIMETYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp"
}


# ==================== CRYPTOGRAPHY ====================

# Hash algorithm
HASH_ALGORITHM = "SHA-256"
HASH_LENGTH = 64  # SHA-256 produces 64 hex characters

# Encryption
ENCRYPTION_ALGORITHM = "AES-256-CBC"
KEY_DERIVATION_ITERATIONS = 100000
SALT_LENGTH = 16  # bytes

# Pseudonym generation
PSEUDONYM_LENGTH = 16  # characters
PSEUDONYM_PREFIX = "anon_"


# ==================== CONSENSUS ====================

# Validator configuration
MIN_VALIDATORS = 15
MAX_VALIDATORS = 20
DEVILS_ADVOCATE_RATIO = 0.1  # 10% devils advocates
CONSENSUS_THRESHOLD = 0.66  # 66% agreement required
VOTING_ROUNDS = 3
CONSENSUS_TIMEOUT = 300  # seconds

# Validator weights
DEFAULT_VALIDATOR_WEIGHT = 1.0
EXPERT_VALIDATOR_WEIGHT = 1.5
DEVILS_ADVOCATE_WEIGHT = 1.0


# ==================== CREDIBILITY ASSESSMENT ====================

# Deepfake detection thresholds
DEEPFAKE_THRESHOLD = 0.5  # Below this = likely fake
HIGH_CONFIDENCE_THRESHOLD = 0.75
LOW_CONFIDENCE_THRESHOLD = 0.25

# Test-time augmentation
NUM_AUGMENTATIONS = 10
AUGMENTATION_THRESHOLD = 0.4  # Entropy threshold for uncertainty

# Cross-modal consistency
MIN_CONSISTENCY_SCORE = 0.6


# ==================== COORDINATION DETECTION ====================

# Similarity thresholds
CONTENT_SIMILARITY_THRESHOLD = 0.7
STYLE_SIMILARITY_THRESHOLD = 0.65
TEMPORAL_SIMILARITY_THRESHOLD = 0.8

# Graph analysis
MIN_CLUSTER_SIZE = 2
MAX_CLUSTER_SIZE = 20
LOUVAIN_RESOLUTION = 1.0

# Anomaly detection
ANOMALY_CONTAMINATION = 0.1  # Expected proportion of outliers
OCSVM_NU = 0.1
OCSVM_GAMMA = "auto"


# ==================== COUNTER-EVIDENCE ====================

# Bayesian aggregation parameters
PRESUMPTION_OF_INNOCENCE_WEIGHT = 1.3  # β in paper
IDENTITY_VERIFICATION_BONUS = 1.2  # Identity verified counter-evidence
PRIOR_CREDIBILITY = 0.5  # Initial credibility before evidence

# Evidence types
EVIDENCE_TYPE_ACCUSATION = "accusation"
EVIDENCE_TYPE_COUNTER = "counter_evidence"
EVIDENCE_TYPE_CORROBORATION = "corroboration"


# ==================== RATE LIMITING ====================

# API rate limits
MAX_SUBMISSIONS_PER_HOUR = 10
MAX_SUBMISSIONS_PER_DAY = 50
MAX_REQUESTS_PER_MINUTE = 60

# IP-based limits
RATE_LIMIT_WINDOW = 3600  # 1 hour in seconds
RATE_LIMIT_CLEANUP_INTERVAL = 300  # 5 minutes


# ==================== TIMEOUTS ====================

# Processing timeouts
SUBMISSION_PROCESSING_TIMEOUT = 600  # 10 minutes
REPORT_GENERATION_TIMEOUT = 120  # 2 minutes
MODEL_INFERENCE_TIMEOUT = 60  # 1 minute

# Cache timeouts
CACHE_TTL = 3600  # 1 hour
SESSION_TTL = 86400  # 24 hours


# ==================== DATABASE/STORAGE ====================

# JSON storage settings
ATOMIC_WRITE_RETRIES = 3
FILE_LOCK_TIMEOUT = 10  # seconds
BACKUP_RETENTION_DAYS = 90

# Indexing
INDEX_REBUILD_INTERVAL = 3600  # 1 hour
INDEX_CACHE_SIZE = 10000


# ==================== SUBMISSION STATUS ====================

class SubmissionStatus(str, Enum):
    """Submission processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


class CredibilityLevel(str, Enum):
    """Credibility assessment levels."""
    HIGH = "high"  # > 0.75
    MEDIUM = "medium"  # 0.4 - 0.75
    LOW = "low"  # < 0.4
    UNSCORED = "unscored"


class EvidenceType(str, Enum):
    """Types of evidence."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    TEXT = "text"


# ==================== REPORTING ====================

# PDF report settings
REPORT_DPI = 300
REPORT_PAGE_SIZE = "A4"
REPORT_FONT_SIZE = 11
REPORT_TITLE_SIZE = 18
REPORT_HEADING_SIZE = 14

# Section 45 compliance
SECTION_45_COMPLIANT = True
INCLUDE_CHAIN_OF_CUSTODY = True
INCLUDE_CONFIDENCE_INTERVALS = True
INCLUDE_ATTENTION_MAPS = True


# ==================== LOGGING ====================

# Log levels
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Log file settings
LOG_FILE_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LOG_FILE_BACKUP_COUNT = 5
LOG_DIR = BASE_DIR / "logs"


# ==================== MONITORING ====================

# Metrics collection
METRICS_ENABLED = True
METRICS_COLLECTION_INTERVAL = 60  # seconds

# Health check
HEALTH_CHECK_INTERVAL = 30  # seconds
HEALTH_CHECK_TIMEOUT = 5  # seconds


# ==================== API SETTINGS ====================

# CORS
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080"
]

# Request/Response
MAX_REQUEST_SIZE = 100 * 1024 * 1024  # 100 MB
REQUEST_TIMEOUT = 300  # 5 minutes
RESPONSE_COMPRESSION = True
COMPRESSION_MIN_SIZE = 1024  # bytes


# ==================== WORKER SETTINGS ====================

# Background processing
WORKER_POOL_SIZE = 4
WORKER_QUEUE_SIZE = 100
WORKER_RETRY_ATTEMPTS = 3
WORKER_RETRY_DELAY = 5  # seconds

# Cleanup
CLEANUP_INTERVAL = 86400  # 24 hours
CLEANUP_AGE_DAYS = 90  # Delete submissions older than 90 days


# ==================== ERROR CODES ====================

class ErrorCode(str, Enum):
    """Application error codes."""
    # General
    INTERNAL_ERROR = "INTERNAL_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    
    # Rate limiting
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # File upload
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    
    # Processing
    PROCESSING_FAILED = "PROCESSING_FAILED"
    MODEL_ERROR = "MODEL_ERROR"
    TIMEOUT = "TIMEOUT"
    
    # Storage
    STORAGE_ERROR = "STORAGE_ERROR"
    INTEGRITY_ERROR = "INTEGRITY_ERROR"


# ==================== HTTP STATUS CODES ====================

HTTP_200_OK = 200
HTTP_201_CREATED = 201
HTTP_400_BAD_REQUEST = 400
HTTP_401_UNAUTHORIZED = 401
HTTP_403_FORBIDDEN = 403
HTTP_404_NOT_FOUND = 404
HTTP_413_PAYLOAD_TOO_LARGE = 413
HTTP_429_TOO_MANY_REQUESTS = 429
HTTP_500_INTERNAL_SERVER_ERROR = 500
HTTP_503_SERVICE_UNAVAILABLE = 503


# ==================== DEVELOPMENT/TESTING ====================

# Feature flags
ENABLE_DEBUGGING = False
ENABLE_PROFILING = False
ENABLE_MOCK_VALIDATORS = False

# Test data
TEST_MODE = False
SEED_VALUE = 42  # For reproducible results


# ==================== PERFORMANCE TUNING ====================

# Memory limits
MAX_MEMORY_USAGE_GB = 8
MODEL_MEMORY_LIMIT_GB = 4

# CPU
MAX_WORKER_THREADS = 4
BATCH_PROCESSING_ENABLED = True

# GPU
USE_GPU_IF_AVAILABLE = True
GPU_MEMORY_FRACTION = 0.8


# ==================== SECURITY ====================

# Authentication (for future implementation)
JWT_SECRET_KEY = "CHANGE_ME_IN_PRODUCTION"  # Must be overridden
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# CSRF protection
CSRF_ENABLED = True

# Input sanitization
MAX_TEXT_LENGTH = 10000  # characters
MAX_NARRATIVE_LENGTH = 5000


# ==================== VALIDATION RULES ====================

# Submission validation
MIN_NARRATIVE_LENGTH = 50  # characters
REQUIRE_EVIDENCE_FILE = True
REQUIRE_LOCATION = False

# Counter-evidence validation
REQUIRE_IDENTITY_VERIFICATION = False
ALLOW_ANONYMOUS_COUNTER = True


# ==================== ENVIRONMENT-SPECIFIC ====================

# Can be overridden by environment variables or config files
import os

# Override from environment
if os.getenv("ENVIRONMENT") == "production":
    ENABLE_DEBUGGING = False
    TEST_MODE = False
    MAX_SUBMISSIONS_PER_HOUR = 5  # Stricter in production

elif os.getenv("ENVIRONMENT") == "development":
    ENABLE_DEBUGGING = True
    LOG_FILE_MAX_BYTES = 1 * 1024 * 1024  # Smaller logs in dev
