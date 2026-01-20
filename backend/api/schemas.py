"""
Pydantic schemas for API request/response validation.
All models include validation rules and documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator, validator, root_validator
from uuid import UUID


# ============================================================================
# ENUMS
# ============================================================================

class EvidenceType(str, Enum):
    """Supported evidence file types."""
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    TEXT = "text"


class SubmissionStatus(str, Enum):
    """Submission processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    DISPUTED = "disputed"
    FAILED = "failed"


class DecisionType(str, Enum):
    """Consensus decision outcomes."""
    FORWARD = "forward"  # Forward to authorities
    REVIEW = "review"    # Human review required
    REJECT = "reject"    # Insufficient evidence


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================

class SubmissionRequest(BaseModel):
    """Request schema for evidence submission."""
    evidence_type: EvidenceType = Field(
        ...,
        description="Type of evidence being submitted"
    )
    text_narrative: Optional[str] = Field(
        None,
        max_length=5000,
        description="Optional text description or narrative"
    )
    metadata: Optional[Dict] = Field(
        default_factory=dict,
        description="Additional metadata (location, date, etc.)"
    )

    @validator('text_narrative')
    def sanitize_text(cls, v):
        """Remove potentially harmful characters."""
        if v:
            # Strip HTML tags and control characters
            import re
            v = re.sub(r'<[^>]+>', '', v)
            v = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', v)
            return v.strip()
        return v

    @validator('metadata')
    def validate_metadata_size(cls, v):
        """Ensure metadata is not excessively large."""
        if v:
            import json
            if len(json.dumps(v)) > 10000:  # 10KB limit
                raise ValueError("Metadata exceeds 10KB limit")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "evidence_type": "image",
                "text_narrative": "Witnessed corruption incident on Jan 10, 2026",
                "metadata": {
                    "location": "City Hall, New Delhi",
                    "incident_date": "2026-01-10"
                }
            }
        }


class CounterEvidenceRequest(BaseModel):
    """Request schema for counter-evidence submission."""
    submission_id: str = Field(
        ...,
        description="UUID of original submission being contested"
    )
    verified_identity: bool = Field(
        default=False,
        description="Whether submitter verified identity via government API"
    )

    @validator('submission_id')
    def validate_uuid(cls, v):
        """Ensure submission_id is valid UUID format."""
        try:
            UUID(v)
            return v
        except ValueError:
            raise ValueError("submission_id must be valid UUID")

    class Config:
        json_schema_extra = {
            "example": {
                "submission_id": "550e8400-e29b-41d4-a716-446655440000",
                "verified_identity": True
            }
        }


class CoordinationGraphQuery(BaseModel):
    """Query parameters for coordination graph retrieval."""
    start_date: Optional[datetime] = Field(
        None,
        description="Filter submissions after this date"
    )
    end_date: Optional[datetime] = Field(
        None,
        description="Filter submissions before this date"
    )
    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum edge similarity threshold"
    )

    @model_validator(mode='after')
    def validate_date_range(cls, values):
        """Ensure end_date is after start_date."""
        start = values.get('start_date')
        end = values.get('end_date')
        if start and end and end < start:
            raise ValueError("end_date must be after start_date")
        return values


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================

class SubmissionResponse(BaseModel):
    """Response after successful evidence submission."""
    submission_id: str = Field(..., description="Unique submission identifier")
    pseudonym: str = Field(..., description="16-character anonymous pseudonym")
    evidence_hash: str = Field(..., description="SHA-256 hash of evidence")
    timestamp: datetime = Field(..., description="Submission timestamp (UTC)")
    status: SubmissionStatus = Field(..., description="Current processing status")
    
    class Config:
        json_schema_extra = {
            "example": {
                "submission_id": "550e8400-e29b-41d4-a716-446655440000",
                "pseudonym": "a3f2b8c9d1e4f5g6",
                "evidence_hash": "5d41402abc4b2a76b9719d911017c592",
                "timestamp": "2026-01-13T06:30:00Z",
                "status": "pending"
            }
        }


class CredibilityScores(BaseModel):
    """Credibility assessment scores."""
    deepfake_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Authenticity score (0=fake, 1=real)"
    )
    consistency_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Cross-modal consistency score"
    )
    plausibility_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Physical plausibility score"
    )
    final_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Weighted aggregate score"
    )
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="90% confidence interval [lower, upper]"
    )

    @validator('confidence_interval')
    def validate_ci(cls, v):
        """Ensure confidence interval bounds are valid."""
        if len(v) != 2:
            raise ValueError("confidence_interval must have exactly 2 values")
        if v[0] > v[1]:
            raise ValueError("Lower bound must be less than upper bound")
        if not (0 <= v[0] <= 1 and 0 <= v[1] <= 1):
            raise ValueError("Confidence bounds must be in [0, 1]")
        return v


class CoordinationInfo(BaseModel):
    """Coordination detection results."""
    flagged: bool = Field(..., description="Whether coordination detected")
    confidence: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Detection confidence"
    )
    community_id: Optional[int] = Field(
        None,
        description="Community cluster ID if flagged"
    )
    similarity_scores: Optional[Dict[str, float]] = Field(
        default_factory=dict,
        description="Similarity to other submissions"
    )


class ConsensusInfo(BaseModel):
    """Byzantine consensus results."""
    votes: Dict[str, int] = Field(
        ...,
        description="Vote counts: {accept, reject}"
    )
    validator_scores: List[float] = Field(
        ...,
        description="Individual validator scores"
    )
    decision: DecisionType = Field(
        ...,
        description="Final consensus decision"
    )
    agreement_percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of validators in agreement"
    )

    @validator('votes')
    def validate_votes(cls, v):
        """Ensure vote counts are non-negative."""
        required_keys = {'accept', 'reject'}
        if not required_keys.issubset(v.keys()):
            raise ValueError(f"votes must contain keys: {required_keys}")
        if any(count < 0 for count in v.values()):
            raise ValueError("Vote counts must be non-negative")
        return v


class CounterEvidenceInfo(BaseModel):
    """Counter-evidence submission details."""
    submitted: bool = Field(..., description="Whether counter-evidence exists")
    counter_evidence_id: Optional[str] = Field(
        None,
        description="UUID of counter-evidence"
    )
    posterior_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Updated score after Bayesian aggregation"
    )
    delta: Optional[float] = Field(
        None,
        description="Score change (posterior - original)"
    )
    identity_verified: Optional[bool] = Field(
        None,
        description="Whether defense verified identity"
    )


class CredibilityResponse(BaseModel):
    """Complete credibility assessment response."""
    submission_id: str
    status: SubmissionStatus
    credibility: Optional[CredibilityScores] = None
    coordination: Optional[CoordinationInfo] = None
    consensus: Optional[ConsensusInfo] = None
    counter_evidence: Optional[CounterEvidenceInfo] = None
    processing_time_seconds: Optional[float] = Field(
        None,
        description="Total processing time"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "submission_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "credibility": {
                    "deepfake_score": 0.82,
                    "consistency_score": 0.75,
                    "plausibility_score": 0.88,
                    "final_score": 0.81,
                    "confidence_interval": [0.73, 0.89]
                },
                "coordination": {
                    "flagged": False,
                    "confidence": 0.65
                },
                "consensus": {
                    "votes": {"accept": 13, "reject": 2},
                    "validator_scores": [0.8, 0.85, 0.78],
                    "decision": "forward",
                    "agreement_percentage": 86.7
                },
                "counter_evidence": {
                    "submitted": False
                },
                "processing_time_seconds": 8.3
            }
        }


class CounterEvidenceResponse(BaseModel):
    """Response after counter-evidence submission."""
    counter_evidence_id: str
    original_submission_id: str
    posterior_score: float = Field(ge=0.0, le=1.0)
    original_score: float = Field(ge=0.0, le=1.0)
    delta: float
    decision_changed: bool
    new_decision: DecisionType
    identity_bonus_applied: bool

    class Config:
        json_schema_extra = {
            "example": {
                "counter_evidence_id": "660e8400-e29b-41d4-a716-446655440111",
                "original_submission_id": "550e8400-e29b-41d4-a716-446655440000",
                "posterior_score": 0.45,
                "original_score": 0.81,
                "delta": -0.36,
                "decision_changed": True,
                "new_decision": "review",
                "identity_bonus_applied": True
            }
        }


class GraphNode(BaseModel):
    """Node in coordination graph."""
    id: str = Field(..., description="Submission UUID")
    label: str = Field(..., description="Pseudonym")
    score: float = Field(..., ge=0.0, le=1.0)
    flagged: bool
    timestamp: datetime


class GraphEdge(BaseModel):
    """Edge in coordination graph."""
    source: str = Field(..., description="Source submission UUID")
    target: str = Field(..., description="Target submission UUID")
    weight: float = Field(..., ge=0.0, le=1.0, description="Similarity weight")
    edge_type: str = Field(..., description="style|time|content")


class Community(BaseModel):
    """Detected community cluster."""
    id: int
    members: List[str] = Field(..., description="List of submission UUIDs")
    modularity: float = Field(..., description="Community modularity score")
    avg_similarity: float = Field(..., ge=0.0, le=1.0)


class CoordinationGraphResponse(BaseModel):
    """Coordination network graph response."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    communities: List[Community]
    total_submissions: int
    flagged_submissions: int


class HealthCheckResponse(BaseModel):
    """System health check response."""
    status: str = Field(..., description="healthy|degraded|unhealthy")
    timestamp: datetime
    checks: Dict[str, bool] = Field(
        ...,
        description="Individual component health status"
    )
    uptime_seconds: float
    version: str = Field(default="1.0.0-mvp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2026-01-13T06:30:00Z",
                "checks": {
                    "storage_readable": True,
                    "storage_writable": True,
                    "models_loaded": True,
                    "memory_ok": True
                },
                "uptime_seconds": 3600.5,
                "version": "1.0.0-mvp"
            }
        }


class MetricsResponse(BaseModel):
    """Performance metrics response."""
    total_submissions: int
    pending_submissions: int
    completed_submissions: int
    failed_submissions: int
    avg_processing_time_seconds: float
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    memory_usage_mb: float
    disk_usage_mb: float


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict] = Field(
        None,
        description="Additional error context"
    )
    timestamp: datetime

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid file type. Only images, audio, and video allowed.",
                "details": {
                    "field": "evidence_type",
                    "received": "document"
                },
                "timestamp": "2026-01-13T06:30:00Z"
            }
        }
