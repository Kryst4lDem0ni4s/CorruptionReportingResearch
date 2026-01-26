#!/usr/bin/env python3
"""
Corruption Reporting System - Full Research Experiment
Version: 1.0.0
Description: Component-level experiment execution script

This script orchestrates the full Corruption Reporting System experiment flow:
1. Setup: Initialize Hashchain Service
2. Layer 1 (Anonymity): Process inputs from provided datasets
3. Layer 2 (Credibility): Run Zero-Shot Inference (CLIP, Wav2Vec2)
4. Layer 3 (Coordination): Group related reports
5. Layer 4 (Consensus): Simulate validator consensus with Byzantine actors
6. Layer 5 (Counter-Evidence): Process conflicting evidence
7. Layer 6 (Reporting): Generate final immutable reports

Usage:
    python execution/run_full_experiment.py
"""

import sys
import json
import time
import random
import logging
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import System Components
from backend.logging_config import setup_logging, get_logger
from backend.services.hash_chain_service import HashChainService
from backend.services.metadata_service import MetadataService
from backend.core.layer2_credibility import Layer2Credibility
from backend.core.layer3_coordination import Layer3Coordination
from backend.core.layer5_counter_evidence import Layer5CounterEvidence
from backend.core.layer6_reporting import Layer6Reporting
from backend.core.layer6_reporting import Layer6Reporting
from evaluation.datasets.dataset_loader import load_dataset, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS

# Real Utils and Services
from backend.utils.text_utils import TextUtils
from backend.utils.graph_utils import GraphUtils
from backend.utils.image_utils import ImageUtils
from backend.utils.audio_utils import AudioUtils
from backend.services.validation_service import ValidationService
from backend.services.crypto_service import CryptoService

# Setup Logging
setup_logging(level="INFO", console=True, file_logging=True)
logger = get_logger("ResearchExperiment")

class ResearchExperiment:
    def __init__(self):
        self.output_dir = project_root / "evaluation" / "results" / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Services
        # Initialize Services
        self.hash_chain = HashChainService(data_dir=self.output_dir / "ledger")
        self.metadata_service = MetadataService()
        self.crypto_service = CryptoService()
        self.validation_service = ValidationService()
        
        # Initialize Utils
        self.image_utils = ImageUtils()
        self.audio_utils = AudioUtils()
        self.text_utils = TextUtils()
        self.graph_utils = GraphUtils()

        # Adapter for Storage (Shared across layers)
        class InMemoryStorage:
            def __init__(self): 
                self.data = {}
                self.submissions = []
            def save_submission(self, data): 
                self.data[data['id']] = data
                # Update list if needed (simple append check)
                existing = next((i for i, s in enumerate(self.submissions) if s['id'] == data['id']), None)
                if existing is not None:
                    self.submissions[existing] = data
                else:
                    self.submissions.append(data)
            def load_submission(self, sub_id): return self.data.get(sub_id)
            def get_all_submissions(self): return self.submissions

        self.storage_adapter = InMemoryStorage()
        
        # Initialize Layers
        self.layer2 = Layer2Credibility(
            storage_service=self.storage_adapter,
            validation_service=self.validation_service,
            image_utils=self.image_utils,
            audio_utils=self.audio_utils,
            crypto_service=self.crypto_service
        )
        
        self.layer3 = Layer3Coordination(
            storage_service=self.storage_adapter,
            text_utils=self.text_utils,
            graph_utils=self.graph_utils,
            min_similarity=0.6,
            time_window_hours=24
        )
        
        self.layer5 = Layer5CounterEvidence(
            storage_service=self.storage_adapter
        )

        self.layer6 = Layer6Reporting(
            storage_service=self.storage_adapter,
            hash_chain_service=self.hash_chain,
            output_dir=self.output_dir / "reports"
        )
        
        self.results = {
            "setup": {"timestamp": datetime.utcnow().isoformat()},
            "layer_stats": {},
            "reports": []
        }

    def run(self):
        logger.info("="*60)
        logger.info("STARTING FULL RESEARCH EXPERIMENT")
        logger.info("="*60)
        
        try:
            # 1. Load Data
            self.dataset = self._load_data()
            
            # 2. Layer 1: Anonymity
            processed_inputs = self._run_layer1_anonymity(self.dataset)
            
            # 3. Layer 2: Credibility
            credibility_reports = self._run_layer2_credibility(processed_inputs)
            
            # 4. Layer 3: Coordination
            grouped_reports = self._run_layer3_coordination(credibility_reports)
            
            # 5. Layer 4: Consensus (Byzantine Simulation)
            consensus_results = self._run_layer4_consensus(grouped_reports)
            
            # 6. Layer 5: Counter-Evidence
            final_set = self._run_layer5_counter_evidence(consensus_results)
            
            # 7. Layer 6: Reporting & Ledger
            self._run_layer6_reporting(final_set)
            
            # Calculate and Add Research Metrics
            self._calculate_metrics(final_set)
            
            # Save final results
            self._save_results()
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            sys.exit(1)

    def _load_data(self) -> List[Dict]:
        logger.info("\n[PHASE 0] Loading Datasets...")
        
        samples = []
        
        # Try loading CelebDF (Video)
        try:
            # We use 'celebdf' which maps to _index_celebdf in loader
            # REDUCED SAMPLES FOR VERIFICATION SPEED
            celeb = load_dataset('celebdf', split='test', max_samples=2)
            samples.extend(celeb['samples'])
            logger.info(f"Loaded {len(celeb['samples'])} samples from CelebDF")
        except Exception as e:
            logger.warning(f"Could not load CelebDF: {e}")
            
        # Try loading Real/Fake Face Detection
        # Check dataset loader logic: 'real-and-fake-face-detection'
        try:
            # Note: The loader might expect the full path name or specific ID
            # Based on file listing, the dir is 'real-and-fake-face-detection'
            # REDUCED SAMPLES FOR VERIFICATION SPEED
            faces = load_dataset('real-and-fake-face-detection', split='test', max_samples=2)
            if not faces['samples']:
                logger.warning("Loaded 0 samples from Face dataset. Check paths.")
            else:
                samples.extend(faces['samples'])
                logger.info(f"Loaded {len(faces['samples'])} samples from Real/Fake Face Detection")
        except Exception as e:
            logger.warning(f"Could not load Face Dataset: {e}")

        # Adding Synthetic Attacks for Layer 3 coordination testing
        try:
            synthetic = load_dataset('synthetic_attacks', max_samples=10) # Load all synthetic
            # Ensure synthetic samples have 'type' set correct for downstream
            for s in synthetic['samples']:
                if 'type' not in s: s['type'] = 'text' # Default fallback
            samples.extend(synthetic['samples'])
            logger.info(f"Loaded {len(synthetic['samples'])} samples from Synthetic Attacks")
        except Exception as e:
            logger.warning(f"Could not load Synthetic Attacks: {e}")

        # [TEST CASE] Cross-Modal Mismatch
        # We take a real image but give it a completely unrelated narrative
        if samples:
            mismatched_sample = samples[0].copy() # Copy a real/fake image sample
            mismatched_sample['text_narrative'] = "This is a video of a flying elephant in the sky."
            mismatched_sample['label'] = 'mismatched_test' 
            # We want to ensure this fails consistency check if Layer 2 is working
            # Note: The loader might not preserve 'text_narrative' automatically, so we'll inject it manually in the loop if needed.
            # But let's add it to the list with a flag we can check in _run_layer2
            mismatched_sample['id_override'] = 'sub_mismatched_001'
            samples.append(mismatched_sample)
            logger.info("Injected [Cross-Modal Mismatch] test case")

        logger.info(f"Total samples loaded: {len(samples)}")
        
        if not samples:
            logger.error("No samples loaded! Experiment cannot proceed.")
            # Create a dummy sample to prevent crash if really nothing found (e.g. CI env)
            logger.info("Injecting dummy sample for pipeline verification.")
            return [{
                'path': str(self.output_dir / 'dummy_evidence.txt'),
                'type': 'text',
                'label': 'real',
                'anonymized_path': str(self.output_dir / 'dummy_evidence.txt')
            }]
            
        return samples

    def _run_layer1_anonymity(self, samples: List[Dict]) -> List[Dict]:
        logger.info("\n[LAYER 1] Anonymity Protocol (Metadata Stripping)...")
        processed = []
        
        for sample in samples:
            try:
                input_path = Path(sample['path'])
                if not input_path.exists():
                    continue
                    
                # Setup output path
                # For experiment, we'll process in-place or copy to temp
                # We use the metadata service methods we just fixed
                
                if sample['type'] == 'image':
                    # MetadataService.strip_image_metadata requires output path
                    output_path = self.output_dir / "anonymized" / input_path.name
                    self.metadata_service.strip_image_metadata(input_path, output_path)
                elif sample['type'] == 'audio':
                    output_path = self.metadata_service.strip_audio_metadata(input_path)
                elif sample['type'] == 'video':
                    output_path = self.metadata_service.strip_video_metadata(input_path)
                else:
                    output_path = input_path # No op for others
                    
                sample['anonymized_path'] = str(output_path)
                processed.append(sample)
                
            except Exception as e:
                logger.warning(f"Failed to anonymize {sample.get('path')}: {e}")
                
        logger.info(f"Anonymized {len(processed)} files.")
        self.results['layer_stats']['layer1_count'] = len(processed)
        return processed

    def _run_layer2_credibility(self, inputs: List[Dict]) -> List[Dict]:
        logger.info("\n[LAYER 2] Credibility Assessment (Zero-Shot Inference)...")
        dataset_reports = []
        
        for item in inputs:
            try:
                evidence_type = item['type'] # 'image', 'video', etc.
                file_path = Path(item['anonymized_path'])
                
                # Special handling for synthetic text attacks
                text_content = item.get('text_narrative', "Experiment test submission")
                
                if evidence_type == 'attack_scenario':
                    evidence_type = 'text'
                    # Try to get text from data if available
                    if 'data' in item:
                        text_content = item['data'].get('text') or item['data'].get('narrative') or text_content
                
                # Determine ID
                if 'id_override' in item:
                    sub_id = item['id_override']
                elif 'attack_id' in item:
                    sub_id = item['attack_id']
                elif 'id' in item:
                    sub_id = item['id']
                else:
                    sub_id = f"sub_{random.randint(1000,9999)}"
                
                # Debug logging for synthetic
                if evidence_type == 'text':
                    logger.info(f"Processing Text Item keys: {item.keys()} -> Assigned ID: {sub_id}")
                
                # Run Layer 2
                # Note: We are using the ACTUAL pre-trained models here via Layer2Credibility
                result = self.layer2.process(
                    submission_id=sub_id,
                    file_path=file_path,
                    evidence_type=evidence_type,
                    text_narrative=text_content, 
                    use_augmentation=False # Speed up experiment
                )
                
                # Attach ground truth if available
                result['ground_truth_label'] = item.get('label')
                
                dataset_reports.append(result)
                logger.info(f"Processed {item['path']} -> Score: {result['final_score']:.4f}")
                
            except Exception as e:
                logger.error(f"Layer 2 failed for {item.get('path')}: {e}")
                
        self.results['layer_stats']['layer2_count'] = len(dataset_reports)
        return dataset_reports

    def _run_layer3_coordination(self, reports: List[Dict]) -> List[Dict]:
        logger.info("\n[LAYER 3] Coordination Analysis...")
        
        processed_reports = []
        for report in reports:
            # Update storage for Layer 3 to query "recent submissions"
            # In live system, this happens async. Here we save first.
            report['id'] = report['submission_id']
            report['timestamp_submission'] = datetime.utcnow().isoformat()
            self.storage_adapter.save_submission(report)
            
            try:
                # Process
                result = self.layer3.process(
                    submission_id=report['submission_id'],
                    text_narrative=report.get('text_narrative'),
                    timestamp=datetime.utcnow()
                )
                
                # Merge results back into report
                report['coordination'] = {
                    'flagged': result['flagged'],
                    'confidence': result['confidence'],
                    'community_size': result.get('community_size', 0)
                }
                
            except Exception as e:
                logger.warning(f"L3 failed for {report['submission_id']}: {e}")
                report['coordination'] = {'flagged': False, 'error': str(e)}
                
            processed_reports.append(report)
            
        self.results['layer_stats']['layer3_count'] = len(processed_reports)
        return processed_reports

    def _run_layer4_consensus(self, reports: List[Dict]) -> List[Dict]:
        logger.info("\n[LAYER 4] Consensus & Byzantine Simulation...")
        # Simulate N validators voting on each report
        
        NUM_VALIDATORS = 5
        BYZANTINE_RATIO = 0.25 # 25% malicious
        
        consensus_results = []
        
        for report in reports:
            votes = []
            for i in range(NUM_VALIDATORS):
                # Honest validator: Votes based on Layer 2 score
                # If Score < 0.5 (Fake) -> Vote REJECT
                # If Score >= 0.5 (Real) -> Vote APPROVE
                if report['final_score'] >= 0.5:
                    vote = "APPROVE"
                else:
                    vote = "REJECT"
                
                # Byzantine Check
                if random.random() < BYZANTINE_RATIO:
                    # Flip vote
                    vote = "REJECT" if vote == "APPROVE" else "APPROVE"
                    
                votes.append(vote)
            
            # Tally
            approve_count = votes.count("APPROVE")
            reject_count = votes.count("REJECT")
            final_decision = "ACCEPTED" if approve_count > reject_count else "REJECTED"
            
            report['consensus'] = {
                'votes': votes,
                'approve': approve_count,
                'reject': reject_count,
                'decision': final_decision,
                'byzantine_simulated': True
            }
            consensus_results.append(report)
            
            logger.info(f"Consensus for {report['submission_id']}: {final_decision} (Approve: {approve_count}, Reject: {reject_count})")

        self.results['layer_stats']['layer4_processed'] = len(consensus_results)
        return consensus_results

    def _run_layer5_counter_evidence(self, reports: List[Dict]) -> List[Dict]:
        logger.info("\n[LAYER 5] Counter-Evidence Simulation...")
        
        finalized = []
        l5_processed_count = 0
        
        for report in reports:
            # Decision logic: Update status first
            decision = report['consensus']['decision']
            if decision == "ACCEPTED":
                # Check ground truth to simulate counter-evidence
                if report.get('ground_truth_label') in ['fake', 'manipulated']:
                    # Simulate Defense Submission
                    logger.info(f"Generating counter-evidence for false positive: {report['submission_id']}")
                    
                    counter_id = f"counter_{report['submission_id']}"
                    # Simulate a "Credible Defense"
                    counter_score = 0.85 # High credibility defense
                    
                    try:
                        # Create minimal counter object in storage for Layer 5 to load
                        counter_obj = {
                            'id': counter_id,
                            'credibility': {'final_score': counter_score},
                            'verified_identity': True,
                            'timestamp_submission': datetime.utcnow().isoformat()
                        }
                        self.storage_adapter.save_submission(counter_obj)
                        
                        # Process via Layer 5
                        l5_result = self.layer5.process(
                            original_submission_id=report['submission_id'],
                            counter_evidence_id=counter_id,
                            counter_credibility_score=counter_score,
                            identity_verified=True
                        )
                        
                        # Update Report
                        report['counter_evidence_challenged'] = True
                        report['counter_evidence_id'] = counter_id
                        report['posterior_score'] = l5_result['posterior_score']
                        report['final_status'] = "DISPUTED" if l5_result['decision_changed'] else "VERIFIED"
                        report['new_decision'] = l5_result['new_decision']
                        
                        l5_processed_count += 1
                        
                    except Exception as e:
                        logger.error(f"L5 failed for {report['submission_id']}: {e}")
                        report['final_status'] = "VERIFIED" # Fallback
                else:
                    report['final_status'] = "VERIFIED"
            else:
                report['final_status'] = "REJECTED"
                
            finalized.append(report)
            
        self.results['layer_stats']['layer5_count'] = l5_processed_count
        return finalized

    def _run_layer6_reporting(self, reports: List[Dict]) -> None:
        logger.info("\n[LAYER 6] Reporting & Ledger Recording...")
        
        for report in reports:
            # Prepare record for storage/reporting
            # Ensure keys match what Layer 6 expects (e.g., 'id', 'credibility', 'consensus')
            if 'credibility' not in report: 
                report['credibility'] = {
                    'final_score': report.get('final_score', 0.0),
                    'deepfake_score': report.get('deepfake_score', 0.0) # Assume these exist or added
                }
            if 'consensus' not in report:
                report['consensus'] = {'decision': report.get('final_status', 'UNKNOWN')}
            
            # 1. Update In-Memory Storage for Layer 6 to read
            # Layer 6 expects 'id', 'pseudonym', etc.
            report['id'] = report['submission_id']
            report['status'] = report['final_status']
            self.storage_adapter.save_submission(report)

            # 2. Generate PDF Report via Layer 6
            try:
                pdf_path = self.layer6.generate_report(report['id'])
                logger.info(f"Generated PDF for {report['id']}")
            except Exception as e:
                logger.error(f"Failed to generate PDF for {report['id']}: {e}")

            # 3. Ledger Recording (Original Logic)
            record = {
                "id": report['submission_id'],
                "evidence_hash": hashlib.sha256(str(report).encode()).hexdigest(),
                "status": report['final_status'],
                "credibility_score": report['final_score'],
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Add to HashChain
            self.hash_chain.add_block(
                submission_id=record['id'],
                evidence_hash=record['evidence_hash'],
                metadata={"status": record['status']}
            )
            report['evidence_hash'] = record['evidence_hash']
            
            self.results['reports'].append(record)
            
        # Get chain stats
        chain_stats = self.hash_chain.get_chain_statistics()
        self.results['ledger_stats'] = chain_stats
        self.results['layer_stats']['layer6_count'] = len(reports)
        logger.info(f"Ledger Updated. Height: {chain_stats['total_blocks']}")

    def _calculate_metrics(self, reports: List[Dict]):
        logger.info("\n[PHASE 4] Calculating Research Metrics...")
        
        y_true = []
        y_scores = []
        y_pred = []
        
        for r in reports:
            # Skip if no ground truth
            if 'ground_truth_label' not in r or not r['ground_truth_label']:
                continue
                
            # Ground Truth: 'fake'/'manipulated' -> 1, 'real' -> 0
            is_fake = 1 if r['ground_truth_label'] in ['fake', 'manipulated'] else 0
            y_true.append(is_fake)
            
            # Score: Layer 2 final score (Deepfake Probability)
            # IMPORTANT: Layer 2 'final_score' is usually "Authenticity Score" (1=Real, 0=Fake)
            # But deepfake detection usually expects Higher Score = Fake.
            # Let's check Layer 2 logic.
            # Layer 2: "Deepfake detection (score < 0.5 typically means fake)" -> So Low = Fake.
            # To get "Probability of Fake", we inverse it: 1 - final_score
            prob_fake = 1.0 - r['final_score']
            y_scores.append(prob_fake)
            
            # Prediction: Based on final status
            # REJECTED -> Detected as Fake (1)
            # DISPUTED -> Detected as Fake (1) (Counter-evidence worked)
            # VERIFIED/ACCEPTED -> Detected as Real (0)
            pred = 1 if r['final_status'] in ['REJECTED', 'DISPUTED'] else 0
            y_pred.append(pred)
            
        if not y_true:
            logger.warning("No ground truth labels found. Skipping metrics.")
            return

        # Calculate Metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        try:
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
                "auroc": roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else None,
                "total_samples": len(y_true),
                "fake_samples": sum(y_true),
                "real_samples": len(y_true) - sum(y_true)
            }
            
            self.results['metrics'] = metrics
            logger.info(f"Metrics Calculated: {json.dumps(metrics, indent=2)}")
            
            # Generate Visualizations
            self._generate_visualizations(y_true, y_scores, y_pred)
            
        except Exception as e:
            logger.error(f"Metrics calculation error: {e}")
            self.results['metrics'] = {"error": str(e)}

    def _generate_visualizations(self, y_true, y_scores, y_pred):
        logger.info("[PHASE 4.1] Generating Visualizations...")
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        try:
            # 1. Score Distribution
            plt.figure(figsize=(10, 6))
            sns.histplot(x=y_scores, hue=y_true, kde=True, bins=20, palette=['green', 'red'])
            plt.title('Credibility Score Distribution (Red=Fake, Green=Real)')
            plt.xlabel('Probability of Deepfake (1 - Credibility Score)')
            plt.ylabel('Count')
            plt.legend(labels=['Fake', 'Real'])
            plt.savefig(figures_dir / "score_distribution.png")
            plt.close()
            
            # 2. Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Predicted Real', 'Predicted Fake'],
                        yticklabels=['Actual Real', 'Actual Fake'])
            plt.title('Confusion Matrix')
            plt.savefig(figures_dir / "confusion_matrix.png")
            plt.close()
            
            # 3. ROC Curve (if mixed classes)
            if len(set(y_true)) > 1:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC)')
                plt.legend(loc="lower right")
                plt.savefig(figures_dir / "roc_curve.png")
                plt.close()
                
            logger.info(f"Visualizations saved to {figures_dir}")
            
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")

    def _save_results(self):
        output_file = self.output_dir / "full_experiment_report.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"\nExperiment Complete! Results saved to: {output_file}")


import hashlib

if __name__ == "__main__":
    experiment = ResearchExperiment()
    experiment.run()
