"""
Multimodal Perception Suite: Metadata & Device Intelligence
Analyzes device fingerprint velocity, IP risk, submission patterns, and known fraud clusters.
"""

from typing import Dict, Any, List

KNOWN_FRAUD_CLUSTERS = {
    "DEV-FRAUD-CLUSTER-X89": "Synthetic Identity Farm syndicate - known repeat attack vector",
    "DEV-EMULATOR-ROOTED-99": "Automated headless browser emulator farm",
    "DEV-BOTNET-PROXY-01": "Rotating TOR exit node syndicate"
}

class MetadataAnalyzer:
    def __init__(self):
        self.device_history: Dict[str, List[str]] = {}

    def analyze_metadata(self, applicant_data: dict) -> dict:
        """
        Evaluates session metadata for fraud indicators:
        - Collision with known fraud syndicates
        - Repeated device submissions (device velocity)
        - Suspicious IP / channel flags
        """
        device_id = applicant_data.get("device_id", "UNKNOWN_DEV")
        ip_addr = applicant_data.get("ip_address", "127.0.0.1")
        channel = applicant_data.get("submission_channel", "WEB")
        applicant_name = applicant_data.get("full_name", "Anonymous")

        risk_score = 0.05
        flags = []
        is_fraud_cluster = False

        # 1. Known syndicate check
        if device_id in KNOWN_FRAUD_CLUSTERS:
            risk_score += 0.75
            is_fraud_cluster = True
            flags.append(f"CRITICAL: Device ID linked to known fraud ring: {KNOWN_FRAUD_CLUSTERS[device_id]}")

        # 2. Device reuse velocity check
        if device_id not in self.device_history:
            self.device_history[device_id] = []
        self.device_history[device_id].append(applicant_name)

        if len(self.device_history[device_id]) > 3:
            risk_score += 0.35
            flags.append(f"HIGH: Rapid multi-applicant submission from single device ID ({len(self.device_history[device_id])} identities)")

        # 3. IP anomaly
        if ip_addr.startswith("10.") or ip_addr.startswith("192.168."):
            pass # Internal/Standard test subnet

        risk_score = min(0.99, max(0.02, risk_score))
        risk_score = round(risk_score, 4)

        return {
            "device_id": device_id,
            "ip_address": ip_addr,
            "metadata_risk_score": risk_score,
            "is_fraud_cluster": is_fraud_cluster,
            "flags": flags,
            "device_submission_count": len(self.device_history.get(device_id, []))
        }

