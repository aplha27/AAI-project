"""
Semantic State & Multimodal Risk Fusion Engine
Implements Baseline (Linear Heuristic) and Improved (Calibrated Logistic & Non-Linear Interaction)
fusion models to fuse vision, acoustic, document, and metadata evidence into an explainable risk score.
"""

import math
from typing import Dict, Any

class RiskFusionEngine:
    def __init__(self):
        # Trained feature weights for calibrated logistic fusion
        self.weights = {
            "face_mismatch": 3.8,      # (1 - face_sim)
            "liveness_spoof": 4.2,     # (1 - liveness_score)
            "text_inconsistency": 3.2, # (1 - consistency_score)
            "doc_blur_glare": 2.1,     # (1 - quality_score)
            "audio_der": 2.9,          # digit error rate
            "audio_replay": 3.6,       # replay_score
            "metadata_risk": 4.5       # metadata risk / fraud cluster
        }
        self.bias = -3.2

    def fuse_baseline(self, perception_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Baseline Method: Simple weighted arithmetic mean of channel risks.
        """
        face_sim = perception_results.get("face_match", {}).get("similarity", 0.5)
        liveness = perception_results.get("liveness", {}).get("liveness_score", 0.5)
        text_cons = perception_results.get("ocr", {}).get("consistency", {}).get("consistency_score", 0.5)
        doc_quality = perception_results.get("ocr", {}).get("quality", {}).get("quality_score", 0.5)
        audio_der = perception_results.get("audio", {}).get("digit_error_rate", 0.0)
        audio_replay = perception_results.get("audio", {}).get("replay_analysis", {}).get("replay_score", 0.0)
        meta_risk = perception_results.get("metadata", {}).get("metadata_risk_score", 0.1)

        # Baseline linear sum
        risk = (
            0.25 * (1.0 - face_sim) +
            0.20 * (1.0 - liveness) +
            0.15 * (1.0 - text_cons) +
            0.10 * (1.0 - doc_quality) +
            0.10 * audio_der +
            0.10 * audio_replay +
            0.10 * meta_risk
        )
        risk = round(float(min(1.0, max(0.0, risk))), 4)

        return {
            "overall_risk_score": risk,
            "fusion_method": "baseline_linear_heuristic",
            "risk_tier": self.get_risk_tier(risk)
        }

    def fuse_improved(self, perception_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Improved Method:
        Calibrated Non-Linear Risk Fusion with Cross-Channel Interaction Terms.
        Accounts for correlated fraud patterns (e.g. screen spoof + audio replay + device reuse).
        """
        face_sim = perception_results.get("face_match", {}).get("similarity", 0.9)
        liveness = perception_results.get("liveness", {}).get("liveness_score", 0.9)
        text_cons = perception_results.get("ocr", {}).get("consistency", {}).get("consistency_score", 0.9)
        doc_quality = perception_results.get("ocr", {}).get("quality", {}).get("quality_score", 0.9)
        audio_der = perception_results.get("audio", {}).get("digit_error_rate", 0.0)
        audio_replay = perception_results.get("audio", {}).get("replay_analysis", {}).get("replay_score", 0.0)
        meta_risk = perception_results.get("metadata", {}).get("metadata_risk_score", 0.05)
        is_fraud_cluster = perception_results.get("metadata", {}).get("is_fraud_cluster", False)

        # Invert genuine metrics to risk metrics (0.0 = safe, 1.0 = high risk)
        f_face = max(0.0, 1.0 - face_sim)
        f_live = max(0.0, 1.0 - liveness)
        f_text = max(0.0, 1.0 - text_cons)
        f_doc = max(0.0, 1.0 - doc_quality)
        f_ader = float(audio_der)
        f_areplay = float(audio_replay)
        f_meta = float(meta_risk)

        # Linear combination
        z = (
            self.bias +
            self.weights["face_mismatch"] * f_face +
            self.weights["liveness_spoof"] * f_live +
            self.weights["text_inconsistency"] * f_text +
            self.weights["doc_blur_glare"] * f_doc +
            self.weights["audio_der"] * f_ader +
            self.weights["audio_replay"] * f_areplay +
            self.weights["metadata_risk"] * f_meta
        )

        # Non-linear Cross-Channel Fraud Interaction Amplification
        # Case A: Combined Presentation Attack (Visual Spoof + Audio Replay)
        if f_live > 0.4 and f_areplay > 0.4:
            z += 2.5

        # Case B: Fraud syndicate collision
        if is_fraud_cluster:
            z += 3.0

        # Case C: Multiple Discrepancies (Face + Text mismatch)
        if f_face > 0.35 and f_text > 0.35:
            z += 2.0

        # Sigmoid Logistic Calibration
        calibrated_risk = 1.0 / (1.0 + math.exp(-z))
        calibrated_risk = round(float(min(0.99, max(0.01, calibrated_risk))), 4)

        risk_tier = self.get_risk_tier(calibrated_risk)

        # Risk breakdown by channel
        breakdown = {
            "face_risk": round(f_face, 4),
            "liveness_risk": round(f_live, 4),
            "document_quality_risk": round(f_doc, 4),
            "text_consistency_risk": round(f_text, 4),
            "voice_challenge_risk": round(max(f_ader, f_areplay), 4),
            "metadata_risk": round(f_meta, 4)
        }

        # Identify primary risk contributors
        primary_factors = []
        if f_face > 0.35:
            primary_factors.append(f"Face similarity below threshold ({face_sim:.2f})")
        if f_live > 0.40:
            primary_factors.append(f"Presentation spoof detected (liveness score {liveness:.2f})")
        if f_doc > 0.40:
            primary_factors.append("Document image degraded by blur or glare")
        if f_text > 0.25:
            primary_factors.append("Discrepancy between application and ID card text")
        if f_areplay > 0.45:
            primary_factors.append("Acoustic replay attack detected in audio challenge")
        if f_ader > 0.0:
            primary_factors.append("Spoken challenge digits did not match required prompt")
        if is_fraud_cluster:
            primary_factors.append("Hardware fingerprint belongs to known fraud syndicate")

        if not primary_factors:
            primary_factors.append("All multimodal perception checks passed cleanly")

        return {
            "overall_risk_score": calibrated_risk,
            "risk_tier": risk_tier,
            "fusion_method": "improved_calibrated_logistic",
            "channel_breakdown": breakdown,
            "primary_factors": primary_factors,
            "is_actionable_escalation": calibrated_risk >= 0.55
        }

    def get_risk_tier(self, score: float) -> str:
        if score < 0.25:
            return "LOW"
        elif score < 0.55:
            return "MEDIUM"
        elif score < 0.80:
            return "HIGH"
        else:
            return "CRITICAL"

    def fuse(self, perception_results: Dict[str, Any], method: str = "improved") -> Dict[str, Any]:
        if method == "baseline":
            return self.fuse_baseline(perception_results)
        return self.fuse_improved(perception_results)

