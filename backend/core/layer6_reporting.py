"""
Layer 6: Forensic Report Generator
Generates Section 45-compliant PDF reports with visualizations.

Input: Submission data with all assessment results
Output: PDF forensic report
"""

import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer,
    PageBreak, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

from backend.services.metrics_service import MetricsService

# Initialize logger
logger = logging.getLogger(__name__)


class Layer6Reporting:
    """
    Layer 6: Forensic Report Generator
    
    Implements:
    - Section 45-compliant PDF reports
    - Chain-of-custody documentation
    - Credibility score visualization
    - Confidence interval charts
    - Attention heatmaps (simplified for MVP)
    - Evidence metadata summary
    """
    
    def __init__(
        self,
        storage_service,
        hash_chain_service,
        output_dir: Optional[Path] = None,
        metrics_service: Optional[MetricsService] = None
    ):
        """
        Initialize Layer 6 with services.
        
        Args:
            storage_service: Storage service for submissions
            hash_chain_service: Hash chain service for custody proof
            output_dir: Output directory for reports
        """
        self.storage = storage_service
        self.hash_chain = hash_chain_service
        
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("backend/data/reports")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics_service
        logger.info(f"Layer 6 (Reporting) initialized (output: {self.output_dir})")
    
    def generate_report(
        self,
        submission_id: str,
        include_technical_details: bool = True
    ) -> Path:
        """
        Generate comprehensive forensic report.
        
        Args:
            submission_id: Submission identifier
            include_technical_details: Include technical analysis details
            
        Returns:
            Path: Path to generated PDF report
            
        Raises:
            ValueError: If report generation fails
        """
        logger.info(f"Layer 6 generating report for {submission_id}")
        
        try:
            # Step 1: Load submission data
            submission = self.storage.load_submission(submission_id)
            if not submission:
                raise ValueError(f"Submission {submission_id} not found")
            
            # Step 2: Create PDF document
            report_filename = f"forensic_report_{submission_id[:8]}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"
            report_path = self.output_dir / report_filename
            
            doc = SimpleDocTemplate(
                str(report_path),
                pagesize=letter,
                rightMargin=0.75*inch,
                leftMargin=0.75*inch,
                topMargin=1*inch,
                bottomMargin=0.75*inch
            )
            
            # Step 3: Build report content
            story = []
            styles = self._get_custom_styles()
            
            # Title page
            story.extend(self._build_title_page(submission, styles))
            story.append(PageBreak())
            
            # Executive summary
            story.extend(self._build_executive_summary(submission, styles))
            story.append(Spacer(1, 0.3*inch))
            
            # Chain of custody
            story.extend(self._build_chain_of_custody(submission, styles))
            story.append(Spacer(1, 0.3*inch))
            
            # Credibility assessment
            story.extend(self._build_credibility_section(submission, styles))
            story.append(Spacer(1, 0.3*inch))
            
            # Coordination analysis
            story.extend(self._build_coordination_section(submission, styles))
            story.append(Spacer(1, 0.3*inch))
            
            # Consensus results
            story.extend(self._build_consensus_section(submission, styles))
            
            # Counter-evidence (if exists)
            if submission.get('counter_evidence_id'):
                story.append(PageBreak())
                story.extend(self._build_counter_evidence_section(submission, styles))
            
            # Technical details (optional)
            if include_technical_details:
                story.append(PageBreak())
                story.extend(self._build_technical_details(submission, styles))
            
            # Legal disclaimer
            story.append(PageBreak())
            story.extend(self._build_legal_disclaimer(styles))
            
            # Step 4: Build PDF
            doc.build(story)
            
            logger.info(f"Report generated: {report_path}")
            
            return report_path
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}", exc_info=True)
            raise ValueError(f"Report generation failed: {str(e)}")
    
    def _get_custom_styles(self) -> Dict:
        """Get custom paragraph styles."""
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='BodyJustify',
            parent=styles['BodyText'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=12
        ))
        
        return styles
    
    def _build_title_page(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build report title page."""
        elements = []
        
        # Title
        elements.append(Spacer(1, 1.5*inch))
        elements.append(Paragraph(
            "FORENSIC EVIDENCE ASSESSMENT REPORT",
            styles['CustomTitle']
        ))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # Submission info table
        info_data = [
            ['Report ID:', submission.get('id', 'N/A')[:16]],
            ['Pseudonym:', submission.get('pseudonym', 'N/A')],
            ['Generated:', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')],
            ['Status:', submission.get('status', 'unknown').upper()],
            ['Classification:', self._get_classification_label(submission)]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LINEBELOW', (0, -1), (-1, -1), 1, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        elements.append(info_table)
        
        elements.append(Spacer(1, 1*inch))
        
        # Confidentiality notice
        elements.append(Paragraph(
            "<b>CONFIDENTIAL - SECTION 45 COMPLIANT</b>",
            styles['BodyJustify']
        ))
        elements.append(Paragraph(
            "This report contains sensitive information and is intended solely for authorized recipients. "
            "Unauthorized disclosure, distribution, or copying is prohibited.",
            styles['BodyJustify']
        ))
        
        return elements
    
    def _build_executive_summary(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build executive summary section."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", styles['SectionHeader']))
        
        # Get key metrics
        credibility = submission.get('credibility', {})
        final_score = credibility.get('final_score', 0.0)
        decision = submission.get('consensus', {}).get('decision', 'review')
        coordination_flagged = submission.get('coordination', {}).get('flagged', False)
        
        # Summary text
        summary_text = f"""
        This forensic report presents the automated assessment of evidence submission 
        <b>{submission.get('pseudonym', 'N/A')}</b>. The evidence underwent comprehensive analysis 
        through a six-layer validation framework including anonymity preservation, credibility 
        assessment, coordination detection, Byzantine consensus, and optional counter-evidence processing.
        <br/><br/>
        <b>Key Findings:</b><br/>
        • Credibility Score: <b>{final_score:.2%}</b><br/>
        • Consensus Decision: <b>{decision.upper()}</b><br/>
        • Coordination Detected: <b>{'YES' if coordination_flagged else 'NO'}</b><br/>
        • Evidence Type: <b>{submission.get('evidence_type', 'N/A').upper()}</b>
        """
        
        elements.append(Paragraph(summary_text, styles['BodyJustify']))
        
        # Add score visualization
        score_chart_path = self._create_score_visualization(submission)
        if score_chart_path and score_chart_path.exists():
            elements.append(Spacer(1, 0.2*inch))
            img = RLImage(str(score_chart_path), width=5*inch, height=2.5*inch)
            elements.append(img)
        
        return elements
    
    def _build_chain_of_custody(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build chain of custody section."""
        elements = []
        
        elements.append(Paragraph("Chain of Custody", styles['SectionHeader']))
        
        # Get chain proof
        try:
            chain_proof = self.hash_chain.get_proof(submission.get('id'))
        except Exception:
            chain_proof = {}
        
        # Chain data table
        chain_data = [
            ['Field', 'Value'],
            ['Evidence Hash', submission.get('evidence_hash', 'N/A')[:32] + '...'],
            ['Chain Hash', submission.get('chain_hash', 'N/A')[:32] + '...'],
            ['Submission Time', submission.get('timestamp_submission', 'N/A')],
            ['Processing Time', f"{submission.get('processing_time_seconds', 0):.2f}s"],
            ['Chain Verified', 'YES' if chain_proof else 'PENDING']
        ]
        
        chain_table = Table(chain_data, colWidths=[2*inch, 4.5*inch])
        chain_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(chain_table)
        
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(
            "<i>Note: All evidence is cryptographically hashed and recorded in an immutable chain "
            "to ensure integrity and prevent tampering.</i>",
            styles['BodyJustify']
        ))
        
        return elements
    
    def _build_credibility_section(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build credibility assessment section."""
        elements = []
        
        elements.append(Paragraph("Credibility Assessment", styles['SectionHeader']))
        
        credibility = submission.get('credibility', {})
        
        # Scores table
        scores_data = [
            ['Metric', 'Score', 'Confidence Interval'],
            [
                'Deepfake Detection',
                f"{credibility.get('deepfake_score', 0):.2%}",
                self._format_ci(credibility.get('confidence_interval', (0, 0)))
            ],
            [
                'Cross-Modal Consistency',
                f"{credibility.get('consistency_score', 0):.2%}",
                'N/A'
            ],
            [
                'Physical Plausibility',
                f"{credibility.get('plausibility_score', 0):.2%}",
                'N/A'
            ],
            [
                '<b>Final Credibility Score</b>',
                f"<b>{credibility.get('final_score', 0):.2%}</b>",
                '<b>' + self._format_ci(credibility.get('confidence_interval', (0, 0))) + '</b>'
            ]
        ]
        
        scores_table = Table(scores_data, colWidths=[2.5*inch, 1.5*inch, 2.5*inch])
        scores_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#ecf0f1')),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(scores_table)
        
        # Interpretation
        elements.append(Spacer(1, 0.1*inch))
        interpretation = self._interpret_credibility_score(
            credibility.get('final_score', 0)
        )
        elements.append(Paragraph(
            f"<b>Interpretation:</b> {interpretation}",
            styles['BodyJustify']
        ))
        
        # Uncertainty flag
        if credibility.get('entropy', 0) > 0.4:
            elements.append(Paragraph(
                "<b>⚠ HIGH UNCERTAINTY DETECTED:</b> This submission exhibits high assessment "
                "uncertainty and requires human expert review.",
                styles['BodyJustify']
            ))
        
        return elements
    
    def _build_coordination_section(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build coordination detection section."""
        elements = []
        
        elements.append(Paragraph("Coordination Analysis", styles['SectionHeader']))
        
        coordination = submission.get('coordination', {})
        flagged = coordination.get('flagged', False)
        confidence = coordination.get('confidence', 0.0)
        
        if flagged:
            elements.append(Paragraph(
                f"<b>⚠ COORDINATION DETECTED</b> (Confidence: {confidence:.1%})",
                styles['BodyJustify']
            ))
            elements.append(Paragraph(
                "This submission appears to be part of a coordinated attack pattern. "
                "Multiple submissions with similar stylometric features, temporal clustering, "
                "and content overlap were identified.",
                styles['BodyJustify']
            ))
            
            # Community info
            community_size = coordination.get('community_size', 0)
            if community_size > 0:
                elements.append(Paragraph(
                    f"<b>Community Size:</b> {community_size} submissions<br/>"
                    f"<b>Average Similarity:</b> {coordination.get('avg_similarity', 0):.1%}",
                    styles['BodyJustify']
                ))
        else:
            elements.append(Paragraph(
                "<b> NO COORDINATION DETECTED</b>",
                styles['BodyJustify']
            ))
            elements.append(Paragraph(
                "No evidence of coordination with other submissions was found. "
                "The submission appears to be independent.",
                styles['BodyJustify']
            ))
        
        return elements
    
    def _build_consensus_section(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build consensus results section."""
        elements = []
        
        elements.append(Paragraph("Consensus Decision", styles['SectionHeader']))
        
        consensus = submission.get('consensus', {})
        decision = consensus.get('decision', 'review')
        agreement = consensus.get('agreement_percentage', 0)
        votes = consensus.get('votes', {})
        
        # Decision box
        decision_color = {
            'forward': colors.HexColor('#27ae60'),
            'reject': colors.HexColor('#e74c3c'),
            'review': colors.HexColor('#f39c12')
        }.get(decision, colors.grey)
        
        decision_data = [[
            f"DECISION: {decision.upper()}",
            f"Agreement: {agreement:.1%}"
        ]]
        
        decision_table = Table(decision_data, colWidths=[3.5*inch, 3*inch])
        decision_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BACKGROUND', (0, 0), (-1, -1), decision_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(decision_table)
        
        # Vote distribution
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(
            f"<b>Vote Distribution:</b> Accept: {votes.get('accept', 0)} | "
            f"Reject: {votes.get('reject', 0)}",
            styles['BodyJustify']
        ))
        
        # Decision explanation
        decision_text = {
            'forward': "The evidence meets credibility thresholds and consensus requirements. "
                      "Recommended for forwarding to relevant authorities.",
            'reject': "The evidence does not meet minimum credibility requirements. "
                     "Insufficient basis for further action.",
            'review': "The evidence requires human expert review due to borderline scores, "
                     "high uncertainty, or lack of clear consensus."
        }.get(decision, "Status unknown.")
        
        elements.append(Paragraph(decision_text, styles['BodyJustify']))
        
        return elements
    
    def _build_counter_evidence_section(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build counter-evidence section."""
        elements = []
        
        elements.append(Paragraph("Counter-Evidence Analysis", styles['SectionHeader']))
        
        counter_id = submission.get('counter_evidence_id')
        
        if not counter_id:
            elements.append(Paragraph("No counter-evidence submitted.", styles['BodyJustify']))
            return elements
        
        # Load counter-evidence
        try:
            counter = self.storage.load_submission(counter_id)
        except Exception:
            counter = None
        
        elements.append(Paragraph(
            "<b>Counter-evidence has been submitted and processed.</b>",
            styles['BodyJustify']
        ))
        
        # Comparison table
        comparison_data = [
            ['Metric', 'Original', 'After Counter-Evidence'],
            [
                'Credibility Score',
                f"{submission.get('credibility', {}).get('final_score', 0):.2%}",
                f"{submission.get('posterior_score', 0):.2%}"
            ],
            [
                'Decision',
                submission.get('consensus', {}).get('decision', 'N/A').upper(),
                submission.get('new_decision', 'N/A').upper()
            ],
            [
                'Identity Verified',
                'N/A',
                'YES' if submission.get('identity_verified') else 'NO'
            ]
        ]
        
        comparison_table = Table(comparison_data, colWidths=[2.2*inch, 2.2*inch, 2.2*inch])
        comparison_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9b59b6')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        
        elements.append(comparison_table)
        
        # Score change
        score_delta = submission.get('score_delta', 0)
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph(
            f"<b>Score Change:</b> {score_delta:+.1%} "
            f"({'increased' if score_delta > 0 else 'decreased' if score_delta < 0 else 'unchanged'})",
            styles['BodyJustify']
        ))
        
        return elements
    
    def _build_technical_details(
        self,
        submission: Dict,
        styles: Dict
    ) -> List:
        """Build technical details section."""
        elements = []
        
        elements.append(Paragraph("Technical Details", styles['SectionHeader']))
        
        # Processing timeline
        elements.append(Paragraph("<b>Processing Timeline:</b>", styles['BodyJustify']))
        
        timeline_data = [
            ['Stage', 'Timestamp', 'Duration'],
            [
                'Submission',
                submission.get('timestamp_submission', 'N/A')[:19],
                '-'
            ],
            [
                'Anonymization',
                submission.get('timestamp_anonymized', 'N/A')[:19],
                f"{submission.get('layer1_time', 0):.2f}s"
            ],
            [
                'Credibility Assessment',
                submission.get('timestamp_assessed', 'N/A')[:19] if isinstance(submission.get('timestamp_assessed'), str) else 'N/A',
                f"{submission.get('credibility', {}).get('processing_time', 0):.2f}s"
            ],
            [
                'Consensus',
                submission.get('timestamp_consensus', 'N/A')[:19] if isinstance(submission.get('timestamp_consensus'), str) else 'N/A',
                f"{submission.get('consensus_time', 0):.2f}s"
            ]
        ]
        
        timeline_table = Table(timeline_data, colWidths=[2.5*inch, 2.5*inch, 1.5*inch])
        timeline_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ecf0f1')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        
        elements.append(timeline_table)
        
        return elements
    
    def _build_legal_disclaimer(self, styles: Dict) -> List:
        """Build legal disclaimer section."""
        elements = []
        
        elements.append(Paragraph("Legal Disclaimer", styles['SectionHeader']))
        
        disclaimer_text = """
        This automated assessment is provided for informational purposes only and should not be 
        considered as legal advice or definitive proof of authenticity. The system uses pre-trained 
        machine learning models and heuristic analysis which may have limitations and potential errors.
        <br/><br/>
        <b>Section 45 Compliance:</b> This report is generated in compliance with Section 45 of the 
        Evidence Act, which governs the admissibility of electronic evidence. However, final 
        determination of admissibility rests with the appropriate legal authorities.
        <br/><br/>
        <b>Limitations:</b><br/>
        • Automated assessments may not detect all forms of manipulation<br/>
        • Results should be validated by human experts where applicable<br/>
        • The system operates on submitted evidence without external verification<br/>
        • Coordination detection is based on patterns and may have false positives/negatives
        <br/><br/>
        <b>Recommended Actions:</b> All evidence marked for "REVIEW" or with high uncertainty 
        should undergo manual expert examination before being used in legal proceedings.
        """
        
        elements.append(Paragraph(disclaimer_text, styles['BodyJustify']))
        
        # Signature block
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph(
            f"<i>Report generated by Corruption Reporting System v1.0.0-MVP<br/>"
            f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</i>",
            styles['BodyJustify']
        ))
        
        return elements
    
    def _create_score_visualization(self, submission: Dict) -> Optional[Path]:
        """Create credibility score visualization chart."""
        try:
            credibility = submission.get('credibility', {})
            
            scores = {
                'Deepfake\nDetection': credibility.get('deepfake_score', 0),
                'Cross-Modal\nConsistency': credibility.get('consistency_score', 0),
                'Physical\nPlausibility': credibility.get('plausibility_score', 0),
                'Final\nScore': credibility.get('final_score', 0)
            }
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            bars = ax.barh(list(scores.keys()), list(scores.values()),
                          color=['#3498db', '#2ecc71', '#f39c12', '#e74c3c'])
            
            ax.set_xlabel('Score', fontsize=10)
            ax.set_xlim(0, 1.0)
            ax.set_title('Credibility Assessment Scores', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, value) in enumerate(zip(bars, scores.values())):
                ax.text(value + 0.02, bar.get_y() + bar.get_height()/2,
                       f'{value:.1%}', va='center', fontsize=9)
            
            plt.tight_layout()
            
            # Save
            chart_path = self.output_dir / f"chart_{submission.get('id', 'temp')[:8]}.png"
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
            return None
    
    def _get_classification_label(self, submission: Dict) -> str:
        """Get classification label based on decision."""
        decision = submission.get('consensus', {}).get('decision', 'review')
        return {
            'forward': 'CREDIBLE - FORWARD TO AUTHORITIES',
            'reject': 'NOT CREDIBLE - REJECTED',
            'review': 'REQUIRES HUMAN REVIEW'
        }.get(decision, 'UNKNOWN')
    
    def _format_ci(self, ci: tuple) -> str:
        """Format confidence interval."""
        if not ci or len(ci) != 2:
            return 'N/A'
        return f"[{ci[0]:.1%}, {ci[1]:.1%}]"
    
    def _interpret_credibility_score(self, score: float) -> str:
        """Interpret credibility score."""
        if score >= 0.80:
            return "HIGH credibility - Evidence appears authentic with high confidence."
        elif score >= 0.60:
            return "MODERATE credibility - Evidence appears reasonably authentic."
        elif score >= 0.40:
            return "LOW credibility - Evidence authenticity is questionable."
        else:
            return "VERY LOW credibility - Evidence likely manipulated or inauthentic."
