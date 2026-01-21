"""
Configuration Loader - Centralized configuration management

Provides:
- YAML configuration loading
- Environment variable overrides
- Multi-environment support (dev, prod)
- Configuration validation
- Runtime configuration updates
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

from backend.constants import (
    BASE_DIR,
    DATA_DIR,
    LOG_DIR,
    DEFAULT_LOG_LEVEL,
    ALLOWED_ORIGINS,
    MAX_REQUEST_SIZE,
    MIN_VALIDATORS,
    MAX_VALIDATORS,
    CONSENSUS_THRESHOLD,
    PRESUMPTION_OF_INNOCENCE_WEIGHT,
    IDENTITY_VERIFICATION_BONUS,
    DEEPFAKE_THRESHOLD,
    NUM_AUGMENTATIONS
)
from backend.logging_config import get_logger
from backend.exceptions import ConfigurationError


logger = get_logger(__name__)


# ==================== CONFIGURATION DATA CLASSES ====================

@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    reload: bool = False
    workers: int = 1
    log_level: str = DEFAULT_LOG_LEVEL
    max_request_size: int = MAX_REQUEST_SIZE
    request_timeout: int = 300
    

@dataclass
class CORSConfig:
    """CORS configuration."""
    allowed_origins: list = field(default_factory=lambda: ALLOWED_ORIGINS)
    allow_credentials: bool = True
    allow_methods: list = field(default_factory=lambda: ["*"])
    allow_headers: list = field(default_factory=lambda: ["*"])


@dataclass
class StorageConfig:
    """Storage configuration."""
    data_dir: Path = DATA_DIR
    submissions_dir: Path = DATA_DIR / "submissions"
    evidence_dir: Path = DATA_DIR / "evidence"
    reports_dir: Path = DATA_DIR / "reports"
    cache_dir: Path = DATA_DIR / "cache"
    backup_retention_days: int = 90
    

@dataclass
class ModelConfig:
    """ML model configuration."""
    use_gpu: bool = True
    use_fp16: bool = True
    batch_size: int = 8
    cache_size: int = 1000
    lazy_loading: bool = True
    model_timeout: int = 60


@dataclass
class ConsensusConfig:
    """Consensus mechanism configuration."""
    min_validators: int = MIN_VALIDATORS
    max_validators: int = MAX_VALIDATORS
    devils_advocate_ratio: float = 0.1
    consensus_threshold: float = CONSENSUS_THRESHOLD
    voting_rounds: int = 3
    timeout: int = 300


@dataclass
class CredibilityConfig:
    """Credibility assessment configuration."""
    deepfake_threshold: float = DEEPFAKE_THRESHOLD
    high_confidence_threshold: float = 0.75
    low_confidence_threshold: float = 0.25
    num_augmentations: int = NUM_AUGMENTATIONS
    min_consistency_score: float = 0.6


@dataclass
class CounterEvidenceConfig:
    """Counter-evidence configuration."""
    presumption_of_innocence_weight: float = PRESUMPTION_OF_INNOCENCE_WEIGHT
    identity_verification_bonus: float = IDENTITY_VERIFICATION_BONUS
    prior_credibility: float = 0.5
    require_identity_verification: bool = False


@dataclass
class CoordinationConfig:
    """Coordination detection configuration."""
    content_similarity_threshold: float = 0.7
    style_similarity_threshold: float = 0.65
    temporal_similarity_threshold: float = 0.8
    min_cluster_size: int = 2
    anomaly_contamination: float = 0.1


@dataclass
class RateLimitConfig:
    """Rate limiting configuration."""
    enabled: bool = True
    max_submissions_per_hour: int = 10
    max_submissions_per_day: int = 50
    max_requests_per_minute: int = 60
    cleanup_interval: int = 300


@dataclass
class SecurityConfig:
    """Security configuration."""
    encryption_enabled: bool = True
    hash_chain_enabled: bool = True
    exif_stripping_enabled: bool = True
    rate_limiting_enabled: bool = True
    csrf_enabled: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = DEFAULT_LOG_LEVEL
    log_dir: Path = LOG_DIR
    console_logging: bool = True
    file_logging: bool = True
    json_logging: bool = False
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class MetricsConfig:
    """Metrics configuration."""
    enabled: bool = True
    collection_interval: int = 60
    export_prometheus: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    environment: str = "development"
    debug: bool = False
    testing: bool = False
    
    server: ServerConfig = field(default_factory=ServerConfig)
    cors: CORSConfig = field(default_factory=CORSConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    credibility: CredibilityConfig = field(default_factory=CredibilityConfig)
    counter_evidence: CounterEvidenceConfig = field(default_factory=CounterEvidenceConfig)
    coordination: CoordinationConfig = field(default_factory=CoordinationConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate()
    
    def _validate(self):
        """Validate configuration values."""
        # Validate consensus
        if not 0 < self.consensus.consensus_threshold <= 1:
            raise ConfigurationError(
                f"Consensus threshold must be between 0 and 1, got {self.consensus.consensus_threshold}",
                config_key="consensus.consensus_threshold"
            )
        
        if self.consensus.min_validators > self.consensus.max_validators:
            raise ConfigurationError(
                f"min_validators ({self.consensus.min_validators}) cannot exceed max_validators ({self.consensus.max_validators})",
                config_key="consensus.validators"
            )
        
        # Validate credibility
        if not 0 < self.credibility.deepfake_threshold <= 1:
            raise ConfigurationError(
                f"Deepfake threshold must be between 0 and 1, got {self.credibility.deepfake_threshold}",
                config_key="credibility.deepfake_threshold"
            )
        
        # Validate counter-evidence
        if self.counter_evidence.presumption_of_innocence_weight < 1.0:
            raise ConfigurationError(
                f"Presumption of innocence weight must be >= 1.0, got {self.counter_evidence.presumption_of_innocence_weight}",
                config_key="counter_evidence.presumption_of_innocence_weight"
            )
        
        # Validate paths exist
        for path_attr in ['data_dir', 'submissions_dir', 'evidence_dir', 'reports_dir', 'cache_dir']:
            path = getattr(self.storage, path_attr)
            if not isinstance(path, Path):
                setattr(self.storage, path_attr, Path(path))


# ==================== CONFIGURATION LOADER ====================

class ConfigLoader:
    """Configuration loader with YAML and environment variable support."""
    
    def __init__(self):
        """Initialize configuration loader."""
        self._config: Optional[AppConfig] = None
    
    def load(
        self, 
        config_file: Optional[str] = None,
        environment: Optional[str] = None
    ) -> AppConfig:
        """
        Load configuration from file and environment.
        
        Args:
            config_file: Path to YAML configuration file
            environment: Environment name (dev, prod)
            
        Returns:
            AppConfig: Loaded configuration
        """
        # Determine environment
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        
        # Load base configuration
        config_data = self._load_yaml(config_file, environment)
        
        # Override with environment variables
        config_data = self._apply_env_overrides(config_data)
        
        # Create configuration object
        self._config = self._build_config(config_data, environment)
        
        logger.info(f"Configuration loaded: environment={environment}")
        
        return self._config
    
    def _load_yaml(
        self,
        config_file: Optional[str],
        environment: str
    ) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            config_file: Path to config file
            environment: Environment name
            
        Returns:
            dict: Configuration data
        """
        # Determine config file path
        if config_file is None:
            # Try environment-specific config first
            env_config_file = BASE_DIR / f"config.{environment}.yaml"
            
            if env_config_file.exists():
                config_file = env_config_file
            else:
                # Fall back to default config
                config_file = BASE_DIR / "config.yaml"
        else:
            config_file = Path(config_file)
        
        # Load YAML if file exists
        if config_file.exists():
            logger.debug(f"Loading configuration from: {config_file}")
            
            try:
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load config file {config_file}: {e}")
                config_data = {}
        else:
            logger.debug(f"Config file not found: {config_file}, using defaults")
            config_data = {}
        
        return config_data
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides.
        
        Environment variables use format: APP_SECTION_KEY
        Example: APP_SERVER_PORT=9000
        
        Args:
            config_data: Base configuration data
            
        Returns:
            dict: Configuration with overrides
        """
        env_prefix = "APP_"
        
        for env_var, value in os.environ.items():
            if not env_var.startswith(env_prefix):
                continue
            
            # Parse environment variable
            # APP_SERVER_PORT -> ['server', 'port']
            parts = env_var[len(env_prefix):].lower().split('_')
            
            if len(parts) < 2:
                continue
            
            section = parts[0]
            key = '_'.join(parts[1:])
            
            # Apply override
            if section not in config_data:
                config_data[section] = {}
            
            # Convert value to appropriate type
            config_data[section][key] = self._convert_value(value)
            
            logger.debug(f"Environment override: {section}.{key} = {value}")
        
        return config_data
    
    def _convert_value(self, value: str) -> Any:
        """
        Convert string value to appropriate type.
        
        Args:
            value: String value
            
        Returns:
            Converted value
        """
        # Boolean
        if value.lower() in ('true', 'yes', '1'):
            return True
        if value.lower() in ('false', 'no', '0'):
            return False
        
        # Integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Float
        try:
            return float(value)
        except ValueError:
            pass
        
        # String
        return value
    
    def _build_config(
        self,
        config_data: Dict[str, Any],
        environment: str
    ) -> AppConfig:
        """
        Build AppConfig from configuration data.
        
        Args:
            config_data: Configuration dictionary
            environment: Environment name
            
        Returns:
            AppConfig: Configuration object
        """
        # Set environment
        config_data['environment'] = environment
        config_data['debug'] = environment == 'development'
        
        # Build nested configurations
        server = ServerConfig(**config_data.get('server', {}))
        cors = CORSConfig(**config_data.get('cors', {}))
        storage = StorageConfig(**config_data.get('storage', {}))
        models = ModelConfig(**config_data.get('models', {}))
        consensus = ConsensusConfig(**config_data.get('consensus', {}))
        credibility = CredibilityConfig(**config_data.get('credibility', {}))
        counter_evidence = CounterEvidenceConfig(**config_data.get('counter_evidence', {}))
        coordination = CoordinationConfig(**config_data.get('coordination', {}))
        rate_limit = RateLimitConfig(**config_data.get('rate_limit', {}))
        security = SecurityConfig(**config_data.get('security', {}))
        logging_config = LoggingConfig(**config_data.get('logging', {}))
        metrics = MetricsConfig(**config_data.get('metrics', {}))
        
        # Create main config
        return AppConfig(
            environment=environment,
            debug=config_data['debug'],
            testing=config_data.get('testing', False),
            server=server,
            cors=cors,
            storage=storage,
            models=models,
            consensus=consensus,
            credibility=credibility,
            counter_evidence=counter_evidence,
            coordination=coordination,
            rate_limit=rate_limit,
            security=security,
            logging=logging_config,
            metrics=metrics
        )
    
    def get_config(self) -> AppConfig:
        """
        Get current configuration.
        
        Returns:
            AppConfig: Current configuration
            
        Raises:
            ConfigurationError: If configuration not loaded
        """
        if self._config is None:
            raise ConfigurationError("Configuration not loaded. Call load() first.")
        
        return self._config


# ==================== GLOBAL CONFIGURATION ====================

# Global configuration loader instance
_config_loader = ConfigLoader()


def load_config(
    config_file: Optional[str] = None,
    environment: Optional[str] = None
) -> AppConfig:
    """
    Load application configuration.
    
    Args:
        config_file: Path to configuration file
        environment: Environment name
        
    Returns:
        AppConfig: Loaded configuration
    """
    return _config_loader.load(config_file, environment)


def get_config() -> AppConfig:
    """
    Get current application configuration.
    
    Returns:
        AppConfig: Current configuration
    """
    return _config_loader.get_config()


def reload_config(
    config_file: Optional[str] = None,
    environment: Optional[str] = None
) -> AppConfig:
    """
    Reload application configuration.
    
    Args:
        config_file: Path to configuration file
        environment: Environment name
        
    Returns:
        AppConfig: Reloaded configuration
    """
    logger.info("Reloading configuration")
    return load_config(config_file, environment)
