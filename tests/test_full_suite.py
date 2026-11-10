"""
Comprehensive Verification & Quality Test Suite (Section 8 of ERP & Deliverable D9)
Verifies:
1. Perception Unit Tests (Face, OCR, Liveness, Audio, Metadata)
2. Risk Fusion Unit Tests (Baseline vs Improved)
3. Policy Store Semantic RAG Retrieval Tests
4. Guardrail & HITL Escalation Trigger Tests (Deliverable D9)
5. End-to-End API Integration & Feedback Loop Tests
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import unittest
import json

from backend.perception.face_matcher import FaceMatcher
from backend.perception.ocr_extractor import OCRExtractor
from backend.perception.liveness_detector import LivenessDetector
from backend.perception.audio_verifier import AudioVerifier
from backend.perception.metadata_analyzer import MetadataAnalyzer
from backend.fusion.risk_fusion import RiskFusionEngine
from backend.agent.rag.policy_store import PolicyVectorStore
from backend.agent.controller import ReActAgentController
from backend.main import app

class TestMultimodalEKycSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets_dir = "backend/data/processed/assets"
        with open("backend/data/processed/bundles_manifest.json") as f:
            cls.bundles = json.load(f)
        cls.bundle_map = {b["bundle_id"]: b for b in cls.bundles}

    # ---------------- 1. PERCEPTION TESTS ----------------

    def test_face_matcher_unit(self):
        fm = FaceMatcher()
        b_clean = self.bundle_map["BUNDLE-001"]
        selfie_p = os.path.join(self.assets_dir, b_clean["verification_assets"]["selfie_image"])
        card_p = os.path.join(self.assets_dir, b_clean["verification_assets"]["id_card_image"])

        res = fm.match(selfie_p, card_p, method="improved")
        self.assertIn("similarity", res)
        self.assertGreaterEqual(res["similarity"], 0.70, "Genuine face pair should exceed 0.70 threshold")
        self.assertTrue(res["match"])

    def test_ocr_and_quality_assessment(self):
        ocr = OCRExtractor()
        b_clean = self.bundle_map["BUNDLE-001"]
        card_p = os.path.join(self.assets_dir, b_clean["verification_assets"]["id_card_image"])

        doc_res = ocr.extract_document_fields(card_p, bundle_metadata=b_clean)
        self.assertTrue(doc_res["success"])
        self.assertIn("quality", doc_res)
        self.assertTrue(doc_res["quality"]["pass_quality_gate"])

        # Consistency
        cons = ocr.verify_consistency(b_clean["applicant_data"], doc_res)
        self.assertTrue(cons["name_match"])
        self.assertTrue(cons["dob_match"])
        self.assertFalse(cons["has_discrepancies"])

    def test_liveness_detector_spoof_vs_live(self):
        ld = LivenessDetector()
        b_live = self.bundle_map["BUNDLE-001"]
        b_spoof = next(b for b in self.bundles if b["category"] == "spoof")

        live_selfie = os.path.join(self.assets_dir, b_live["verification_assets"]["selfie_image"])
        spoof_selfie = os.path.join(self.assets_dir, b_spoof["verification_assets"]["selfie_image"])

        res_live = ld.detect(live_selfie, method="improved")
        res_spoof = ld.detect(spoof_selfie, method="improved")

        self.assertGreaterEqual(res_live["liveness_score"], 0.65, "Genuine selfie must pass liveness")
        self.assertLess(res_spoof["liveness_score"], 0.50, "Spoof presentation must trigger liveness alert")
        self.assertTrue(res_spoof["device_bezel_detected"] or res_spoof["screen_raster_detected"])

    def test_audio_asr_and_replay_detection(self):
        av = AudioVerifier()
        b_live = self.bundle_map["BUNDLE-001"]
        b_spoof = next(b for b in self.bundles if b["category"] == "spoof")

        audio_live = os.path.join(self.assets_dir, b_live["verification_assets"]["audio_challenge"])
        audio_spoof = os.path.join(self.assets_dir, b_spoof["verification_assets"]["audio_challenge"])

        res_live = av.decode_spoken_digits(audio_live, b_live["verification_assets"]["expected_challenge_digits"], bundle_metadata=b_live)
        res_spoof = av.decode_spoken_digits(audio_spoof, b_spoof["verification_assets"]["expected_challenge_digits"], bundle_metadata=b_spoof)

        self.assertEqual(res_live["digit_error_rate"], 0.0)
        self.assertFalse(res_live["replay_analysis"]["is_replay"])
        self.assertTrue(res_spoof["replay_analysis"]["is_replay"])

    def test_metadata_analyzer_fraud_cluster(self):
        ma = MetadataAnalyzer()
        safe_applicant = {"device_id": "DEV-CLEAN-01", "ip_address": "192.168.1.10"}
        fraud_applicant = {"device_id": "DEV-FRAUD-CLUSTER-X89", "ip_address": "192.168.1.50"}

        safe_res = ma.analyze_metadata(safe_applicant)
        fraud_res = ma.analyze_metadata(fraud_applicant)

        self.assertFalse(safe_res["is_fraud_cluster"])
        self.assertTrue(fraud_res["is_fraud_cluster"])
        self.assertGreater(fraud_res["metadata_risk_score"], 0.70)

    # ---------------- 2. RISK FUSION TESTS ----------------

    def test_risk_fusion_calibration(self):
        engine = RiskFusionEngine()
        clean_bundle = {
            "face_match": {"similarity": 0.92},
            "ocr": {"quality": {"quality_score": 0.95}, "consistency": {"consistency_score": 1.0}},
            "liveness": {"liveness_score": 0.95},
            "audio": {"digit_error_rate": 0.0, "replay_analysis": {"replay_score": 0.02}},
            "metadata": {"metadata_risk_score": 0.05, "is_fraud_cluster": False}
        }
        res_clean = engine.fuse(clean_bundle, method="improved")
        self.assertEqual(res_clean["risk_tier"], "LOW")
        self.assertLess(res_clean["overall_risk_score"], 0.25)

        attack_bundle = {
            "face_match": {"similarity": 0.50},
            "ocr": {"quality": {"quality_score": 0.95}, "consistency": {"consistency_score": 1.0}},
            "liveness": {"liveness_score": 0.10},
            "audio": {"digit_error_rate": 0.5, "replay_analysis": {"replay_score": 0.92}},
            "metadata": {"metadata_risk_score": 0.85, "is_fraud_cluster": True}
        }
        res_attack = engine.fuse(attack_bundle, method="improved")
        self.assertIn(res_attack["risk_tier"], ["HIGH", "CRITICAL"])
        self.assertGreater(res_attack["overall_risk_score"], 0.80)

    # ---------------- 3. RAG POLICY RETRIEVAL TESTS ----------------

    def test_policy_rag_retrieval(self):
        store = PolicyVectorStore()
        # Query 1: Optical quality
        results_blur = store.query_policy("Document blurry glare image quality retry", top_k=1)
        self.assertEqual(results_blur[0]["policy_id"], "POL-001")

        # Query 2: Liveness presentation attack
        results_spoof = store.query_policy("Screen moire raster spoof presentation attack", top_k=1)
        self.assertEqual(results_spoof[0]["policy_id"], "POL-002")

        # Query 3: Face similarity
        results_face = store.query_policy("Facial match cosine similarity threshold", top_k=1)
        self.assertEqual(results_face[0]["policy_id"], "POL-003")

    # ---------------- 4. GUARDRAIL & HITL ESCALATION TESTS (D9) ----------------

    def test_guardrail_optical_quality_retry(self):
        controller = ReActAgentController()
        b_blur = next(b for b in self.bundles if b["category"] == "genuine_poor_capture")
        res = controller.run_investigation(b_blur, self.assets_dir)

        self.assertEqual(res["final_decision"], "RETRY")
        self.assertEqual(res["guardrail_triggered"], "GUARDRAIL_OPTICAL_QUALITY_RETRY")
        self.assertEqual(res["total_steps"], 2, "Early stopping should halt further compute after quality failure")

    def test_guardrail_presentation_attack_reject(self):
        controller = ReActAgentController()
        b_spoof = next(b for b in self.bundles if b["category"] == "spoof")
        res = controller.run_investigation(b_spoof, self.assets_dir)

        self.assertEqual(res["final_decision"], "REJECT")
        self.assertEqual(res["guardrail_triggered"], "GUARDRAIL_PRESENTATION_ATTACK_DEFENSE")
        self.assertTrue(res["escalated_to_hitl"], "Spoof must mandate human supervisor escalation")

    def test_guardrail_demographic_mismatch(self):
        controller = ReActAgentController()
        b_mismatch = next(b for b in self.bundles if b["category"] == "mismatch")
        res = controller.run_investigation(b_mismatch, self.assets_dir)

        self.assertEqual(res["final_decision"], "MANUAL_REVIEW")
        self.assertEqual(res["guardrail_triggered"], "GUARDRAIL_IDENTITY_DISCREPANCY")
        self.assertTrue(res["escalated_to_hitl"])

    # ---------------- 5. END-TO-END API & HITL FEEDBACK TESTS ----------------

    def test_api_health_endpoint(self):
        from backend.main import health_check
        data = health_check()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertIn("face_matcher", data["models_loaded"])

    def test_api_verify_and_trace(self):
        from backend.main import verify_identity, VerificationRequest
        req = VerificationRequest(bundle_id="BUNDLE-001")
        data = verify_identity(req)
        self.assertEqual(data["case_id"], "BUNDLE-001")
        self.assertEqual(data["final_decision"], "APPROVE")
        self.assertIn("action_trace", data)
        self.assertGreater(len(data["action_trace"]), 0)

    def test_api_hitl_review_feedback_loop(self):
        from backend.main import verify_identity, VerificationRequest, get_reviewer_queue, submit_human_review, ReviewSubmission
        # 1. Post verification for a flagged spoof case
        b_spoof = next(b for b in self.bundles if b["category"] == "spoof")
        verify_identity(VerificationRequest(bundle_id=b_spoof["bundle_id"]))

        # 2. Check reviewer queue
        queue = get_reviewer_queue()
        self.assertTrue(any(c["case_id"] == b_spoof["bundle_id"] for c in queue))

        # 3. Submit human review verdict
        sub = ReviewSubmission(
            reviewer_id="REV-TEST-09",
            verdict="REJECT",
            notes="Presentation attack confirmed via moire lines.",
            override_reason="Supervisor sign-off on automated quarantine."
        )
        rev_data = submit_human_review(b_spoof["bundle_id"], sub)
        self.assertEqual(rev_data["status"], "REVIEW_RECORDED")

        # 4. Verify case is resolved in reviewer queue
        q_after = get_reviewer_queue()
        self.assertFalse(any(c["case_id"] == b_spoof["bundle_id"] for c in q_after))

if __name__ == "__main__":
    unittest.main()
