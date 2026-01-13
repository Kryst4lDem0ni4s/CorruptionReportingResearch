/**
 * Corruption Reporting System - Credibility Display Logic
 * Version: 1.0.0
 * Description: Handles submission review, status polling, and result display
 * 
 * This module manages:
 * - Loading submission data
 * - Status polling
 * - Credibility score display
 * - Coordination detection results
 * - Consensus results
 * - Report generation
 * - Counter-evidence navigation
 * 
 * Dependencies: api.js, utils.js (optional)
 */

// ============================================
// GLOBAL STATE
// ============================================
let currentSubmission = null;
let pollingInterval = null;
let submissionId = null;

// ============================================
// CONFIGURATION
// ============================================
const CONFIG = {
    pollingInterval: 3000, // 3 seconds
    maxPollingTime: 300000, // 5 minutes
    autoRefresh: true
};

// ============================================
// INITIALIZATION
// ============================================

/**
 * Initialize the review page
 */
function initReviewPage() {
    console.log('Initializing review page...');
    
    // Get submission ID from URL
    submissionId = getSubmissionIdFromURL();
    
    if (!submissionId) {
        showSearchForm();
        setupSearchForm();
    } else {
        loadSubmission(submissionId);
    }
    
    // Setup action buttons
    setupActionButtons();
    
    console.log('Review page initialized');
}

/**
 * Get submission ID from URL parameters
 * @returns {string|null} Submission ID
 */
function getSubmissionIdFromURL() {
    const params = new URLSearchParams(window.location.search);
    return params.get('id');
}

/**
 * Setup search form
 */
function setupSearchForm() {
    const searchForm = document.getElementById('search-form');
    const searchInput = document.getElementById('submission-id-input');
    const searchButton = document.getElementById('search-button');
    
    if (!searchForm) return;
    
    searchForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const id = searchInput?.value.trim();
        if (id) {
            // Update URL and load
            window.history.pushState({}, '', `?id=${id}`);
            loadSubmission(id);
        }
    });
    
    // Enable search on input
    if (searchInput) {
        searchInput.addEventListener('input', () => {
            if (searchButton) {
                searchButton.disabled = !searchInput.value.trim();
            }
        });
    }
}

/**
 * Setup action buttons
 */
function setupActionButtons() {
    const refreshButton = document.getElementById('refresh-btn');
    const downloadButton = document.getElementById('download-report-btn');
    const counterButton = document.getElementById('submit-counter-btn');
    
    if (refreshButton) {
        refreshButton.addEventListener('click', () => {
            if (submissionId) {
                loadSubmission(submissionId, false); // Don't start polling
            }
        });
    }
    
    if (downloadButton) {
        downloadButton.addEventListener('click', downloadReport);
    }
    
    if (counterButton) {
        counterButton.addEventListener('click', () => {
            if (submissionId) {
                window.location.href = `/counter.html?id=${submissionId}`;
            }
        });
    }
}

// ============================================
// DATA LOADING
// ============================================

/**
 * Load submission data
 * @param {string} id - Submission ID
 * @param {boolean} startPolling - Whether to start status polling
 */
async function loadSubmission(id, startPolling = true) {
    submissionId = id;
    
    // Show loading state
    showLoading();
    hideSearchForm();
    
    try {
        console.log(`Loading submission: ${id}`);
        const submission = await api.getSubmission(id);
        
        currentSubmission = submission;
        
        // Display submission data
        displaySubmission(submission);
        
        // Start polling if processing
        if (startPolling && (submission.status === 'pending' || submission.status === 'processing')) {
            startStatusPolling();
        }
        
    } catch (error) {
        console.error('Failed to load submission:', error);
        showError(getErrorMessage(error));
    } finally {
        hideLoading();
    }
}

/**
 * Start status polling
 */
function startStatusPolling() {
    // Stop any existing polling
    stopStatusPolling();
    
    console.log('Starting status polling...');
    
    pollingInterval = setInterval(async () => {
        try {
            const submission = await api.getSubmission(submissionId);
            currentSubmission = submission;
            displaySubmission(submission);
            
            // Stop polling if complete or failed
            if (submission.status === 'completed' || submission.status === 'failed') {
                stopStatusPolling();
            }
        } catch (error) {
            console.error('Polling error:', error);
            // Continue polling despite errors
        }
    }, CONFIG.pollingInterval);
    
    // Auto-stop after max time
    setTimeout(() => {
        stopStatusPolling();
    }, CONFIG.maxPollingTime);
}

/**
 * Stop status polling
 */
function stopStatusPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log('Stopped status polling');
    }
}

// ============================================
// DISPLAY FUNCTIONS
// ============================================

/**
 * Display submission data
 * @param {Object} submission - Submission data
 */
function displaySubmission(submission) {
    // Show details section
    const detailsSection = document.getElementById('submission-details');
    if (detailsSection) {
        detailsSection.classList.remove('d-none');
    }
    
    // Display basic info
    displayBasicInfo(submission);
    
    // Display status
    displayStatus(submission);
    
    // Display processing progress
    if (submission.status === 'processing') {
        displayProgress(submission);
    } else {
        hideProgress();
    }
    
    // Display results if available
    if (submission.status === 'completed') {
        displayResults(submission);
    }
    
    // Display error if failed
    if (submission.status === 'failed') {
        displayFailure(submission);
    }
}

/**
 * Display basic submission info
 * @param {Object} submission - Submission data
 */
function displayBasicInfo(submission) {
    // Submission ID
    const idElement = document.getElementById('display-submission-id');
    if (idElement) {
        idElement.textContent = submission.submission_id || 'N/A';
    }
    
    // Pseudonym
    const pseudonymElement = document.getElementById('display-pseudonym');
    if (pseudonymElement) {
        pseudonymElement.textContent = submission.pseudonym || 'N/A';
    }
    
    // Timestamp
    const timestampElement = document.getElementById('display-timestamp');
    if (timestampElement) {
        const date = new Date(submission.timestamp);
        timestampElement.textContent = date.toLocaleString();
    }
    
    // Location
    const locationElement = document.getElementById('display-location');
    if (locationElement) {
        locationElement.textContent = submission.location || 'Not specified';
    }
    
    // File count
    const fileCountElement = document.getElementById('display-file-count');
    if (fileCountElement) {
        const count = (submission.evidence_files || []).length;
        fileCountElement.textContent = count;
    }
}

/**
 * Display status
 * @param {Object} submission - Submission data
 */
function displayStatus(submission) {
    const statusElement = document.getElementById('display-status');
    const statusBadge = document.getElementById('status-badge');
    const statusHeader = document.getElementById('status-header');
    
    const status = submission.status || 'unknown';
    
    if (statusElement) {
        statusElement.textContent = status.charAt(0).toUpperCase() + status.slice(1);
    }
    
    // Update badge
    if (statusBadge) {
        statusBadge.textContent = status.toUpperCase();
        statusBadge.className = `badge badge-${status}`;
    }
    
    // Update header color
    if (statusHeader) {
        statusHeader.className = `card-header status-${status}`;
    }
}

/**
 * Display processing progress
 * @param {Object} submission - Submission data
 */
function displayProgress(submission) {
    const progressSection = document.getElementById('processing-progress-section');
    const progressBar = document.getElementById('processing-progress');
    const progressDesc = document.getElementById('progress-description');
    const stageElement = document.getElementById('display-stage');
    
    if (progressSection) {
        progressSection.classList.remove('d-none');
    }
    
    const stage = submission.processing_stage || 'unknown';
    const progress = getProgressFromStage(stage);
    
    if (progressBar) {
        progressBar.style.width = `${progress}%`;
        progressBar.textContent = `${progress}%`;
        progressBar.className = `progress-bar progress-stage-${Math.ceil(progress / 20)}`;
    }
    
    if (progressDesc) {
        progressDesc.textContent = getStageDescription(stage);
    }
    
    if (stageElement) {
        stageElement.textContent = stage.charAt(0).toUpperCase() + stage.slice(1);
    }
}

/**
 * Hide progress section
 */
function hideProgress() {
    const progressSection = document.getElementById('processing-progress-section');
    if (progressSection) {
        progressSection.classList.add('d-none');
    }
}

/**
 * Display results
 * @param {Object} submission - Submission data
 */
function displayResults(submission) {
    // Display credibility assessment
    displayCredibility(submission);
    
    // Display coordination detection
    displayCoordination(submission);
    
    // Display consensus
    displayConsensus(submission);
    
    // Enable download button
    const downloadButton = document.getElementById('download-report-btn');
    if (downloadButton) {
        downloadButton.disabled = false;
    }
}

/**
 * Display credibility assessment
 * @param {Object} submission - Submission data
 */
function displayCredibility(submission) {
    const credibilityCard = document.getElementById('credibility-card');
    if (!credibilityCard) return;
    
    credibilityCard.classList.remove('d-none');
    
    const credibility = submission.credibility_assessment || {};
    const score = credibility.overall_score || 0;
    
    // Overall score
    const scoreElement = document.getElementById('credibility-score');
    if (scoreElement) {
        scoreElement.textContent = (score * 100).toFixed(1) + '%';
        scoreElement.className = `display-4 credibility-score score-${getScoreLevel(score)}`;
    }
    
    // Credibility bar
    const barElement = document.getElementById('credibility-bar');
    if (barElement) {
        barElement.style.width = `${score * 100}%`;
        barElement.className = `progress-bar credibility-${getScoreLevel(score)}`;
    }
    
    // Credibility level
    const levelElement = document.getElementById('credibility-level');
    if (levelElement) {
        levelElement.textContent = getCredibilityLevel(score);
        levelElement.className = `fw-bold level-${getScoreLevel(score)}`;
    }
    
    // Deepfake detection
    const deepfakeScore = document.getElementById('deepfake-score');
    const deepfakeConfidence = document.getElementById('deepfake-confidence');
    
    if (deepfakeScore && credibility.deepfake_score !== undefined) {
        const isAuthentic = credibility.deepfake_score > 0.5;
        deepfakeScore.textContent = isAuthentic ? 'Likely Authentic' : 'Suspicious';
        deepfakeScore.className = isAuthentic ? 'text-success fw-bold' : 'text-danger fw-bold';
    }
    
    if (deepfakeConfidence && credibility.deepfake_confidence !== undefined) {
        deepfakeConfidence.textContent = (credibility.deepfake_confidence * 100).toFixed(1) + '%';
    }
}

/**
 * Display coordination detection
 * @param {Object} submission - Submission data
 */
function displayCoordination(submission) {
    const coordinationCard = document.getElementById('coordination-card');
    if (!coordinationCard) return;
    
    coordinationCard.classList.remove('d-none');
    
    const coordination = submission.coordination_detection || {};
    const risk = coordination.risk_level || 'none';
    
    // Risk level
    const riskElement = document.getElementById('coordination-risk');
    if (riskElement) {
        riskElement.textContent = risk.charAt(0).toUpperCase() + risk.slice(1) + ' Risk';
        riskElement.className = `fw-bold risk-${risk}`;
    }
    
    // Show warning if high risk
    const warningElement = document.getElementById('coordination-warning');
    if (warningElement) {
        if (risk === 'high') {
            warningElement.classList.remove('d-none');
        } else {
            warningElement.classList.add('d-none');
        }
    }
}

/**
 * Display consensus results
 * @param {Object} submission - Submission data
 */
function displayConsensus(submission) {
    const consensusCard = document.getElementById('consensus-card');
    if (!consensusCard) return;
    
    consensusCard.classList.remove('d-none');
    
    const consensus = submission.consensus || {};
    
    // Agreement percentage
    const agreementElement = document.getElementById('consensus-agreement');
    if (agreementElement && consensus.agreement !== undefined) {
        const agreement = consensus.agreement * 100;
        agreementElement.textContent = agreement.toFixed(1) + '%';
        agreementElement.className = `display-4 agreement-${getAgreementLevel(consensus.agreement)}`;
    }
    
    // Validator count
    const validatorElement = document.getElementById('validator-count');
    if (validatorElement && consensus.validator_count) {
        validatorElement.textContent = consensus.validator_count;
    }
    
    // Consensus status
    const statusElement = document.getElementById('consensus-status');
    if (statusElement && consensus.status) {
        statusElement.textContent = consensus.status;
    }
}

/**
 * Display failure message
 * @param {Object} submission - Submission data
 */
function displayFailure(submission) {
    const errorState = document.getElementById('error-state');
    const errorMessage = document.getElementById('error-message');
    
    if (errorState) {
        errorState.classList.remove('d-none');
    }
    
    if (errorMessage) {
        errorMessage.textContent = submission.error_message || 'Processing failed';
    }
}

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Get progress percentage from processing stage
 * @param {string} stage - Processing stage
 * @returns {number} Progress percentage
 */
function getProgressFromStage(stage) {
    const stages = {
        'anonymization': 20,
        'credibility_assessment': 40,
        'coordination_detection': 60,
        'consensus': 80,
        'complete': 100
    };
    return stages[stage] || 0;
}

/**
 * Get stage description
 * @param {string} stage - Processing stage
 * @returns {string} Description
 */
function getStageDescription(stage) {
    const descriptions = {
        'anonymization': 'Anonymizing submission...',
        'credibility_assessment': 'Analyzing evidence credibility...',
        'coordination_detection': 'Detecting coordination patterns...',
        'consensus': 'Building consensus...',
        'complete': 'Processing complete'
    };
    return descriptions[stage] || 'Processing...';
}

/**
 * Get score level
 * @param {number} score - Score (0-1)
 * @returns {string} Level (low/medium/high)
 */
function getScoreLevel(score) {
    if (score >= 0.75) return 'high';
    if (score >= 0.5) return 'medium';
    return 'low';
}

/**
 * Get credibility level text
 * @param {number} score - Score (0-1)
 * @returns {string} Credibility level
 */
function getCredibilityLevel(score) {
    if (score >= 0.75) return 'High Credibility';
    if (score >= 0.5) return 'Medium Credibility';
    return 'Low Credibility';
}

/**
 * Get agreement level
 * @param {number} agreement - Agreement (0-1)
 * @returns {string} Level (low/medium/high)
 */
function getAgreementLevel(agreement) {
    if (agreement >= 0.7) return 'high';
    if (agreement >= 0.5) return 'medium';
    return 'low';
}

// ============================================
// UI ACTIONS
// ============================================

/**
 * Download report
 */
async function downloadReport() {
    if (!submissionId) {
        showError('No submission ID');
        return;
    }
    
    const downloadButton = document.getElementById('download-report-btn');
    
    try {
        if (downloadButton) {
            downloadButton.disabled = true;
            downloadButton.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Downloading...';
        }
        
        await api.downloadReportFile(submissionId, `report_${submissionId}.pdf`);
        
        console.log('Report downloaded successfully');
        
    } catch (error) {
        console.error('Failed to download report:', error);
        showError(getErrorMessage(error));
    } finally {
        if (downloadButton) {
            downloadButton.disabled = false;
            downloadButton.innerHTML = '<i class="bi bi-download me-2"></i>Download Report';
        }
    }
}

// ============================================
// UI STATE MANAGEMENT
// ============================================

/**
 * Show loading state
 */
function showLoading() {
    const loadingState = document.getElementById('loading-state');
    if (loadingState) {
        loadingState.classList.remove('d-none');
    }
}

/**
 * Hide loading state
 */
function hideLoading() {
    const loadingState = document.getElementById('loading-state');
    if (loadingState) {
        loadingState.classList.add('d-none');
    }
}

/**
 * Show search form
 */
function showSearchForm() {
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        searchForm.classList.remove('d-none');
    }
}

/**
 * Hide search form
 */
function hideSearchForm() {
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        searchForm.classList.add('d-none');
    }
}

/**
 * Show error message
 * @param {string} message - Error message
 */
function showError(message) {
    const errorState = document.getElementById('error-state');
    const errorMessage = document.getElementById('error-message');
    
    if (errorState) {
        errorState.classList.remove('d-none');
    }
    
    if (errorMessage) {
        errorMessage.textContent = message;
    }
}

// ============================================
// CLEANUP
// ============================================

/**
 * Cleanup on page unload
 */
function cleanup() {
    stopStatusPolling();
}

// ============================================
// INITIALIZATION ON PAGE LOAD
// ============================================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initReviewPage);
} else {
    initReviewPage();
}

// Cleanup on page unload
window.addEventListener('beforeunload', cleanup);

// Export for external use (if needed)
if (typeof window !== 'undefined') {
    window.ReviewHandler = {
        init: initReviewPage,
        load: loadSubmission,
        refresh: () => loadSubmission(submissionId, false),
        downloadReport,
        currentSubmission,
        cleanup
    };
}
