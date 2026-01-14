"""
Core Module - Business Logic Layers

This module contains the 6-layer framework implementation:
- Layer 1: Anonymous Submission Gateway
- Layer 2: Credibility Assessment Engine
- Layer 3: Coordination Detection System
- Layer 4: Byzantine Consensus Simulator
- Layer 5: Counter-Evidence Processor
- Layer 6: Forensic Report Generator

Plus the Orchestrator that coordinates all layers.
"""

from backend.core.layer1_anonymity import Layer1Anonymity
from backend.core.layer2_credibility import Layer2Credibility
from backend.core.layer3_coordination import Layer3Coordination
from backend.core.layer4_consensus import Layer4Consensus
from backend.core.layer5_counter_evidence import Layer5CounterEvidence
from backend.core.layer6_reporting import Layer6Reporting
from backend.core.orchestrator import Orchestrator
from backend.services.metrics_service import MetricsService

__all__ = [
    'Layer1Anonymity',
    'Layer2Credibility',
    'Layer3Coordination',
    'Layer4Consensus',
    'Layer5CounterEvidence',
    'Layer6Reporting',
    'Orchestrator'
]

__version__ = '1.0.0-MVP'
