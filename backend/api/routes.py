"""
FastAPI route definitions for all API endpoints.
Implements 6-layer architecture with proper error handling.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4
import traceback
import json
import asyncio


from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
    BackgroundTasks
)
from fastapi.responses import FileResponse, StreamingResponse

from backend.api import schemas
from backend.api.dependencies import (
    get_rate_limiter,
    get_storage_service,
    get_orchestrator,
    verify_submission_exists
)
from backend.services.validation_service import ValidationService

# Initialize logger
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


# ============================================================================
# EVIDENCE SUBMISSION ENDPOINT
# ============================================================================

@router.post(
    "/submissions",
    response_model=schemas.SubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit evidence for credibility assessment",
    description="""
    Submit evidence (image/audio/video) for anonymous credibility assessment.
    Processing happens asynchronously - poll /submissions/{id} for results.
    
    **Rate Limit:** 1000 submissions per hour per IP address.
    **File Limits:** Images/Audio ≤5MB, Video ≤50MB.
    """,
    responses={
        202: {"description": "Submission accepted for processing"},
        400: {"description": "Invalid request or file type"},
        413: {"description": "File size exceeds limits"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal processing error"}
    }
)
async def submit_evidence(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Evidence file to analyze"),
    evidence_type: str = Form(..., description="Type: image|audio|video|text"),
    text_narrative: Optional[str] = Form(None, description="Optional narrative (max 5000 chars)"),
    metadata: Optional[str] = Form(None, description="Optional JSON metadata"),
    rate_limiter=Depends(get_rate_limiter),
    storage_service=Depends(get_storage_service),
    orchestrator=Depends(get_orchestrator)
):
    """
    Submit evidence for anonymous credibility assessment.
    Returns immediately with submission ID for status polling.
    """
    start_time = time.time()
    
    try:
        # Parse and validate request
        metadata_dict = json.loads(metadata) if metadata else {}
        
        request_data = schemas.SubmissionRequest(
            evidence_type=evidence_type,
            text_narrative=text_narrative,
            metadata=metadata_dict
        )
        
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid JSON in metadata field"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Validate file
    validation_service = ValidationService()
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file size and type
        # Note: validate_file_content requires (filename, content, evidence_type)
        is_valid, error_msg = validation_service.validate_file_content(
            filename=file.filename,
            content=file_content,
            evidence_type=request_data.evidence_type
        )
        
        if not is_valid:
            raise ValueError(error_msg)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"File validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to validate uploaded file"
        )
    
    # Generate submission ID
    submission_id = str(uuid4())
    
    try:
        # Save file to staging area
        staging_path = storage_service.save_evidence_file(
            submission_id=submission_id,
            filename=file.filename,
            content=file_content,
            evidence_type=request_data.evidence_type
        )
        
        # Create initial submission record
        submission_data = {
            "id": submission_id,
            "evidence_type": request_data.evidence_type.value,
            "text_narrative": request_data.text_narrative,
            "metadata": request_data.metadata,
            "timestamp_submission": datetime.utcnow().isoformat(),
            "status": schemas.SubmissionStatus.PENDING.value,
            "file_path": str(staging_path)
        }
        
        storage_service.save_submission(submission_id, submission_data)
        
        # Queue background processing
        background_tasks.add_task(
            process_submission_async,
            submission_id=submission_id,
            file_path=staging_path,
            evidence_type=request_data.evidence_type.value,
            orchestrator=orchestrator,
            storage_service=storage_service
        )
        
        logger.info(
            f"Submission {submission_id} queued for processing "
            f"(took {time.time() - start_time:.2f}s)"
        )
        
        # Return immediate response
        return schemas.SubmissionResponse(
            submission_id=submission_id,
            pseudonym=submission_data.get("pseudonym", "pending"),
            evidence_hash=submission_data.get("evidence_hash", "pending"),
            timestamp=datetime.fromisoformat(submission_data["timestamp_submission"]),
            status=schemas.SubmissionStatus.PENDING
        )
        
    except Exception as e:
        logger.error(f"Submission creation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process submission: {str(e)}"
        )


# ============================================================================
# CREDIBILITY STATUS ENDPOINT
# ============================================================================

@router.get(
    "/submissions/{submission_id}",
    response_model=schemas.CredibilityResponse,
    summary="Get submission status and results",
    description="""
    Retrieve current status and results for a submission.
    Poll this endpoint every 5 seconds until status is 'completed'.
    """,
    responses={
        200: {"description": "Submission details retrieved"},
        404: {"description": "Submission not found"},
        500: {"description": "Retrieval error"}
    }
)
async def get_submission(
    submission_id: str,
    storage_service=Depends(get_storage_service),
    _=Depends(verify_submission_exists)
):
    """
    Get credibility assessment results for a submission.
    Returns current processing status and all available results.
    """
    try:
        # Load submission data
        submission = storage_service.load_submission(submission_id)
        
        if not submission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Submission {submission_id} not found"
            )
        
        # Parse status
        status_value = schemas.SubmissionStatus(submission.get("status", "pending"))
        
        # Build response
        response = schemas.CredibilityResponse(
            submission_id=submission_id,
            status=status_value
        )
        
        # Add credibility scores if available
        if "credibility" in submission and submission["credibility"]:
            cred = submission["credibility"]
            response.credibility = schemas.CredibilityScores(
                deepfake_score=cred["deepfake_score"],
                consistency_score=cred["consistency_score"],
                plausibility_score=cred["plausibility_score"],
                final_score=cred["final_score"],
                confidence_interval=tuple(cred["confidence_interval"])
            )
        
        # Add coordination info if available
        if "coordination" in submission and submission["coordination"]:
            coord = submission["coordination"]
            response.coordination = schemas.CoordinationInfo(
                flagged=coord["flagged"],
                confidence=coord.get("confidence", 0.0),
                community_id=coord.get("community_id"),
                similarity_scores=coord.get("similarity_scores", {})
            )
        
        # Add consensus info if available
        if "consensus" in submission and submission["consensus"]:
            cons = submission["consensus"]
            response.consensus = schemas.ConsensusInfo(
                votes=cons["votes"],
                validator_scores=cons["validator_scores"],
                decision=schemas.DecisionType(cons["decision"]),
                agreement_percentage=cons.get("agreement_percentage", 0.0)
            )
        
        # Add counter-evidence info if available
        counter_id = submission.get("counter_evidence_id")
        if counter_id:
            response.counter_evidence = schemas.CounterEvidenceInfo(
                submitted=True,
                counter_evidence_id=counter_id,
                posterior_score=submission.get("posterior_score"),
                delta=submission.get("score_delta"),
                identity_verified=submission.get("identity_verified", False)
            )
        else:
            response.counter_evidence = schemas.CounterEvidenceInfo(
                submitted=False
            )
        
        # Add processing time if completed
        if status_value == schemas.SubmissionStatus.COMPLETED:
            response.processing_time_seconds = submission.get("processing_time_seconds")
        
        if status_value == schemas.SubmissionStatus.FAILED:
            response.error = submission.get("error")
            response.error_traceback = submission.get("error_traceback")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving credibility for {submission_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve submission details"
        )


# ============================================================================
# COUNTER-EVIDENCE ENDPOINT
# ============================================================================

@router.post(
    "/counter-evidence",
    response_model=schemas.CounterEvidenceResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit counter-evidence to contest accusation",
    description="""
    Submit defense evidence to contest an original accusation.
    Uses Bayesian aggregation with 1.3× presumption-of-innocence weighting.
    
    **Identity Verification:** If verified_identity=true, applies 1.2× bonus.
    """,
    responses={
        201: {"description": "Counter-evidence processed"},
        400: {"description": "Invalid request"},
        404: {"description": "Original submission not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Processing error"}
    }
)
async def submit_counter_evidence(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    submission_id: str = Form(..., description="Original submission UUID"),
    verified_identity: bool = Form(False, description="Identity verified flag"),
    storage_service=Depends(get_storage_service),
    orchestrator=Depends(get_orchestrator),
    rate_limiter=Depends(get_rate_limiter)
):
    """
    Submit counter-evidence to defend against accusation.
    Processes through same pipeline, then applies Bayesian aggregation.
    """
    try:
        # Validate request
        request_data = schemas.CounterEvidenceRequest(
            submission_id=submission_id,
            verified_identity=verified_identity
        )
        
        # Check original submission exists
        original = storage_service.load_submission(submission_id)
        if not original:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Original submission {submission_id} not found"
            )
        
        # Check if counter-evidence already submitted
        if original.get("counter_evidence_id"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Counter-evidence already submitted for this accusation"
            )
        
        # Validate and save counter-evidence file
        file_content = await file.read()
        validation_service = ValidationService()
        
        # Determine evidence type from original submission
        evidence_type = original.get("evidence_type", "image")
        
        validation_service.validate_file_upload(
            filename=file.filename,
            content=file_content,
            evidence_type=evidence_type
        )
        
        # Generate counter-evidence ID
        counter_id = str(uuid4())
        
        # Save counter-evidence file
        staging_path = storage_service.save_evidence_file(
            submission_id=counter_id,
            filename=file.filename,
            content=file_content,
            evidence_type=evidence_type
        )
        
        # Create counter-evidence submission
        counter_data = {
            "id": counter_id,
            "original_submission_id": submission_id,
            "evidence_type": evidence_type,
            "verified_identity": verified_identity,
            "timestamp_submission": datetime.utcnow().isoformat(),
            "status": schemas.SubmissionStatus.PENDING.value,
            "file_path": str(staging_path)
        }
        
        storage_service.save_submission(counter_id, counter_data)
        
        # Queue background processing
        background_tasks.add_task(
            process_counter_evidence_async,
            counter_evidence_id=counter_id,
            original_submission_id=submission_id,
            file_path=staging_path,
            verified_identity=verified_identity,
            orchestrator=orchestrator,
            storage_service=storage_service
        )
        
        logger.info(f"Counter-evidence {counter_id} submitted for {submission_id}")
        
        # Return immediate response (final scores calculated async)
        return schemas.CounterEvidenceResponse(
            counter_evidence_id=counter_id,
            original_submission_id=submission_id,
            posterior_score=0.0,  # Will be updated after processing
            original_score=original.get("credibility", {}).get("final_score", 0.0),
            delta=0.0,
            decision_changed=False,
            new_decision=schemas.DecisionType(original.get("consensus", {}).get("decision", "review")),
            identity_bonus_applied=verified_identity
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Counter-evidence submission failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process counter-evidence: {str(e)}"
        )


# ============================================================================
# REPORT GENERATION ENDPOINT
# ============================================================================

@router.get(
    "/report/{submission_id}",
    response_class=FileResponse,
    summary="Download PDF forensic report",
    description="""
    Generate and download Section 45-compliant forensic report.
    Includes credibility scores, chain-of-custody, and attention visualizations.
    """,
    responses={
        200: {
            "description": "PDF report",
            "content": {"application/pdf": {}}
        },
        404: {"description": "Submission not found or not completed"},
        500: {"description": "Report generation error"}
    }
)
async def download_report(
    submission_id: str,
    storage_service=Depends(get_storage_service),
    orchestrator=Depends(get_orchestrator),
    _=Depends(verify_submission_exists)
):
    """
    Generate and download PDF forensic report for a submission.
    Report includes all assessment results and legal disclaimers.
    """
    try:
        # Load submission
        submission = storage_service.load_submission(submission_id)
        
        if not submission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Submission {submission_id} not found"
            )
        
        # Check if processing completed
        if submission.get("status") != schemas.SubmissionStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Report generation only available for completed submissions"
            )
        
        # Generate report (via Layer 6)
        report_path = await orchestrator.generate_report(submission_id)
        
        if not report_path or not Path(report_path).exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to generate report"
            )
        
        logger.info(f"Report generated for submission {submission_id}")
        
        # Return PDF file
        return FileResponse(
            path=report_path,
            media_type="application/pdf",
            filename=f"forensic_report_{submission_id[:8]}.pdf",
            headers={
                "Content-Disposition": f"attachment; filename=report_{submission_id[:8]}.pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report generation failed for {submission_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate report"
        )


# ============================================================================
# COORDINATION GRAPH ENDPOINT
# ============================================================================

@router.post(
    "/coordination/detect",
    response_model=schemas.CoordinationDetectionResponse,
    summary="Check for coordination among submissions",
    description="Analyze a set of submissions for potential coordination patterns."
)
async def detect_coordination(
    request: schemas.CoordinationRequest,
    rate_limiter = Depends(get_rate_limiter),
    orchestrator = Depends(get_orchestrator)
):
    """
    Detect coordination in a set of submissions.
    Checks if the provided submission IDs form or belong to a coordinated cluster.
    """
    try:
        # Get current coordination graph
        graph_data = orchestrator.layer3.get_coordination_graph_data()
        
        communities = graph_data.get('communities', [])
        target_ids = set(request.submission_ids)
        
        is_coordinated = False
        confidence = 0.0
        details = []
        
        # Check if target IDs appear in any detected community
        for community in communities:
            members = set(community.get('members', []))
            
            # Calculate overlap
            overlap = target_ids.intersection(members)
            
            if len(overlap) >= 2: # At least 2 submissions in same community
                # Calculate what % of the check set is in this community
                coverage = len(overlap) / len(target_ids)
                
                # Calculate what % of community is the check set
                community_coverage = len(overlap) / len(members)
                
                if coverage > 0.5 or community_coverage > 0.5:
                    is_coordinated = True
                    # Confidence based on overlap and community size
                    confidence = max(confidence, (coverage + community_coverage) / 2)
                    details.append(community)
        
        return {
            "is_coordinated": is_coordinated,
            "confidence": confidence,
            "clusters": details,
            "metrics": {
                "total_checked": len(target_ids),
                "communities_found": len(details)
            }
        }
        
    except Exception as e:
        logger.error(f"Coordination detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Coordination detection failed: {str(e)}"
        )

@router.get(
    "/coordination-graph",
    response_model=schemas.CoordinationGraphResponse,
    summary="Get coordination detection network graph",
    description="""
    Retrieve network graph showing similarity relationships between submissions.
    Useful for visualizing coordinated attack patterns.
    """,
    responses={
        200: {"description": "Graph data retrieved"},
        500: {"description": "Graph generation error"}
    }
)
async def get_coordination_graph(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    min_similarity: float = 0.7,
    storage_service=Depends(get_storage_service),
    orchestrator=Depends(get_orchestrator)
):
    """
    Get coordination detection graph with nodes (submissions) and edges (similarities).
    Filters by date range and minimum similarity threshold.
    """
    try:
        # Validate query parameters
        query = schemas.CoordinationGraphQuery(
            start_date=start_date,
            end_date=end_date,
            min_similarity=min_similarity
        )
        
        # Generate graph data (via Layer 3)
        graph_data = await orchestrator.get_coordination_graph(
            start_date=query.start_date,
            end_date=query.end_date,
            min_similarity=query.min_similarity
        )
        
        # Parse into response schema
        nodes = [
            schemas.GraphNode(
                id=node["id"],
                label=node["label"],
                score=node["score"],
                flagged=node["flagged"],
                timestamp=datetime.fromisoformat(node["timestamp"])
            )
            for node in graph_data.get("nodes", [])
        ]
        
        edges = [
            schemas.GraphEdge(
                source=edge["source"],
                target=edge["target"],
                weight=edge["weight"],
                edge_type=edge["type"]
            )
            for edge in graph_data.get("edges", [])
        ]
        
        communities = [
            schemas.Community(
                id=comm["id"],
                members=comm["members"],
                modularity=comm["modularity"],
                avg_similarity=comm.get("avg_similarity", 0.0)
            )
            for comm in graph_data.get("communities", [])
        ]
        
        return schemas.CoordinationGraphResponse(
            nodes=nodes,
            edges=edges,
            communities=communities,
            total_submissions=len(nodes),
            flagged_submissions=sum(1 for n in nodes if n.flagged)
        )
        
    except Exception as e:
        logger.error(f"Coordination graph generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate coordination graph"
        )


# ============================================================================
# BACKGROUND PROCESSING FUNCTIONS
# ============================================================================

async def process_submission_async(
    submission_id: str,
    file_path: Path,
    evidence_type: str,
    orchestrator,
    storage_service
):
    """
    Background task: Process submission through all 6 layers.
    Updates submission status as processing progresses.
    """
    start_time = time.time()

    try:
        logger.info(f"🚀 Background processing started for {submission_id}")
        
        # Load existing submission data to preserve it
        existing_submission = storage_service.load_submission(submission_id)
        if not existing_submission:
            raise ValueError(f"Submission {submission_id} not found in storage")

        # Update status to processing (preserve existing fields)
        existing_submission.update({
            'status': 'processing',
            'progress': 10,
            'timestamp_started': datetime.utcnow().isoformat()
        })
        storage_service.save_submission(submission_id, existing_submission)
        
        logger.info(f"Processing submission {submission_id} through orchestrator")
        logger.debug(f"File path: {file_path}, Evidence type: {evidence_type}")
        logger.debug(f"File exists: {file_path.exists() if isinstance(file_path, Path) else 'Not a Path'}")

        # Execute full 6-layer pipeline
        # Note: orchestrator.process_submission is async and returns Dict
        results = await orchestrator.process_submission(
            submission_id=submission_id,
            file_path=file_path,
            evidence_type=evidence_type
        )

        # Calculate processing time
        processing_time = time.time() - start_time
        results["processing_time_seconds"] = processing_time

        # Update submission with results (preserve all existing fields)
        results['status'] = 'completed'
        results['progress'] = 100
        results['timestamp_completed'] = datetime.utcnow().isoformat()
        
        # Merge with existing submission data
        existing_submission.update(results)
        storage_service.save_submission(submission_id, existing_submission)

        logger.info(
            f"✅ Submission {submission_id} completed in {processing_time:.2f}s"
        )

    except Exception as e:
        error_msg = str(e)
        error_trace = traceback.format_exc()
        processing_time = time.time() - start_time

        logger.error(
            f"❌ Processing failed for {submission_id}: {error_msg}",
            exc_info=True
        )
        logger.error(f"Full error traceback:\n{error_trace}")

        # Update status to failed with detailed error
        try:
            # Load existing submission data to preserve all fields
            existing_submission = storage_service.load_submission(submission_id)
            if not existing_submission:
                existing_submission = {'id': submission_id}
            
            # Update with error information (preserve all existing fields)
            existing_submission.update({
                'status': 'failed',
                'progress': 0,
                'error': error_msg,
                'error_traceback': error_trace,
                'processing_time_seconds': processing_time,
                'timestamp_failed': datetime.utcnow().isoformat()
            })
            
            storage_service.save_submission(submission_id, existing_submission)
            logger.info(f"✅ Error details saved successfully for {submission_id}")
            
        except Exception as update_error:
            logger.error(f"❌ CRITICAL: Failed to save error details: {update_error}")
            logger.error(f"Update error traceback:\n{traceback.format_exc()}")



async def process_counter_evidence(
    counter_evidence_id: str,
    original_submission_id: str,
    file_path: Path,
    verified_identity: bool,
    orchestrator,
    storage_service
):
    """
    Background task: Process counter-evidence and perform Bayesian aggregation.
    """
    try:
        logger.info(f"Processing counter-evidence {counter_evidence_id}")
        
        # Process counter-evidence through layers 1-4
        counter_results = await orchestrator.process_submission(
            submission_id=counter_evidence_id,
            file_path=file_path,
            evidence_type="image" # Default
        )
        
        # Step 2: Get counter credibility score
        counter_score = counter_results.get('credibility', {}).get('final_score', 0.5)
        
        # Step 3: Bayesian aggregation
        aggregation_results = await orchestrator.process_counter_evidence(
            original_submission_id=original_submission_id,
            counter_evidence_id=counter_evidence_id,
            counter_credibility_score=counter_score,
            identity_verified=verified_identity
        )
        
        # Update counter-evidence submission status
        if hasattr(storage_service, 'update_submission'):
            storage_service.update_submission(
                counter_evidence_id, 
                {'status': 'completed', 'results': counter_results}
            )
        else:
            storage_service.save_submission_status(counter_evidence_id, 'completed')
        
        logger.info(f"Counter-evidence aggregation completed")
        
    except Exception as e:
        logger.error(f"Counter-evidence processing failed: {e}", exc_info=True)
