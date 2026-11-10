"""
KYC Policy Corpus, Vector Store & Semantic RAG Engine
Maintains regulatory runbooks (RBI Master Direction / FATF / eKYC standards)
and provides grounded semantic policy retrieval for the ReAct controller.
"""

import math
from typing import List, Dict, Any

KYC_POLICY_CORPUS = [
    {
        "policy_id": "POL-001",
        "title": "Document Optical Quality and Capture Standards",
        "category": "DOCUMENT_QUALITY",
        "text": (
            "Rule POL-001.1: If document image sharpness (Laplacian variance) is below 110.0 or specular glare "
            "exceeds 4.5% of total card area, OCR field extraction cannot be guaranteed. "
            "Rule POL-001.2: If candidate submission attempt count is less than 3, the agent MUST issue a RETRY action, "
            "instructing the user to disable camera flash and provide a steady, well-lit photo. "
            "Rule POL-001.3: If attempt count reaches or exceeds 3, the case must be escalated to MANUAL_REVIEW."
        ),
        "keywords": ["blur", "glare", "quality", "retry", "sharpness", "recapture", "attempt"]
    },
    {
        "policy_id": "POL-002",
        "title": "Biometric Presentation Attack Detection (PAD)",
        "category": "LIVENESS_SECURITY",
        "text": (
            "Rule POL-002.1: Straight-through auto-approval requires a calibrated facial liveness score >= 0.65. "
            "Rule POL-002.2: If high-frequency FFT moiré raster patterns, screen borders, or device bezels are detected "
            "(liveness < 0.40), the agent MUST flag a Presentation Attack. "
            "Rule POL-002.3: In accordance with bank anti-fraud standards, presentation attacks MUST be escalated to "
            "Human-in-the-loop (HITL) supervisor for immediate REJECT logging. Never auto-approve."
        ),
        "keywords": ["liveness", "spoof", "moire", "screen", "pad", "attack", "bezel", "deepfake"]
    },
    {
        "policy_id": "POL-003",
        "title": "Facial Biometric Match and Impersonation Defense",
        "category": "BIOMETRIC_MATCH",
        "text": (
            "Rule POL-003.1: Straight-through approval requires facial cosine similarity between selfie and ID card >= 0.70. "
            "Rule POL-003.2: Borderline similarity (0.60 <= score < 0.70) under acceptable lighting requires secondary human "
            "biometric review. "
            "Rule POL-003.3: Facial similarity below 0.60 with crisp document quality constitutes probable identity impersonation "
            "and MUST be escalated to MANUAL_REVIEW or REJECT."
        ),
        "keywords": ["face", "match", "similarity", "impersonation", "photo", "selfie", "threshold"]
    },
    {
        "policy_id": "POL-004",
        "title": "Demographic Consistency and Name Discrepancy Tolerance",
        "category": "DATA_INTEGRITY",
        "text": (
            "Rule POL-004.1: Date of Birth (DOB) between application form and government credential must match exactly. "
            "Rule POL-004.2: Name matching allows a fuzzy similarity tolerance of >= 0.85 to accommodate minor transliteration "
            "variations or missing middle initials. "
            "Rule POL-004.3: Any discrepancy in DOB or name similarity below 0.85 prohibits automatic approval and requires "
            "escalation to MANUAL_REVIEW."
        ),
        "keywords": ["name", "dob", "birth", "discrepancy", "consistency", "mismatch", "fuzzy"]
    },
    {
        "policy_id": "POL-005",
        "title": "Acoustic Challenge & Voice Anti-Spoofing Protocols",
        "category": "AUDIO_CHALLENGE",
        "text": (
            "Rule POL-005.1: The applicant must speak the exact 4-digit dynamic challenge sequence prompted on screen. "
            "Rule POL-005.2: If Digit Error Rate (DER) > 0, the spoken challenge fails. "
            "Rule POL-005.3: If audio spectral flatness or speaker resonance harmonics indicate acoustic replay playback "
            "(replay score >= 0.50), the transaction must be flagged as a synthetic replay attack and sent for REJECT via HITL."
        ),
        "keywords": ["audio", "voice", "challenge", "digits", "replay", "acoustic", "spoken", "speaker"]
    },
    {
        "policy_id": "POL-006",
        "title": "Device Fingerprinting and Fraud Syndicate Velocity",
        "category": "DEVICE_FRAUD",
        "text": (
            "Rule POL-006.1: Submission hardware fingerprints matching known fraudulent device clusters or bot farms "
            "(e.g., DEV-FRAUD-CLUSTER) MUST be immediately quarantined. "
            "Rule POL-006.2: Velocity exceeding 3 distinct identities from a single hardware device within 24 hours "
            "triggers an automatic syndication alert and mandates REJECT with fraud incident logging."
        ),
        "keywords": ["device", "syndicate", "cluster", "velocity", "hardware", "ip", "fraud", "botnet"]
    }
]

class PolicyVectorStore:
    """
    Semantic RAG Retriever over KYC Runbooks and Regulatory Directives.
    """
    def __init__(self):
        self.corpus = KYC_POLICY_CORPUS
        self.vocabulary = self._build_vocab()

    def _build_vocab(self):
        vocab = set()
        for doc in self.corpus:
            words = doc["text"].lower().split() + doc["keywords"]
            for w in words:
                cleaned = "".join(c for c in w if c.isalnum())
                if len(cleaned) > 2:
                    vocab.add(cleaned)
        return sorted(list(vocab))

    def _embed_text(self, text: str) -> List[float]:
        words = text.lower().split()
        cleaned_words = ["".join(c for c in w if c.isalnum()) for w in words]
        counts = {}
        for w in cleaned_words:
            if w in self.vocabulary:
                counts[w] = counts.get(w, 0) + 1

        vec = [float(counts.get(term, 0)) for term in self.vocabulary]
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def query_policy(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most relevant policy rules matching the query.
        """
        q_vec = self._embed_text(query)

        scored = []
        for doc in self.corpus:
            d_vec = self._embed_text(doc["text"] + " " + " ".join(doc["keywords"]))
            dot_product = sum(a * b for a, b in zip(q_vec, d_vec))
            # Boost if keywords appear directly
            q_lower = query.lower()
            keyword_hits = sum(1 for kw in doc["keywords"] if kw in q_lower)
            final_score = dot_product + (0.25 * keyword_hits)
            scored.append((final_score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in scored[:top_k]:
            results.append({
                "policy_id": doc["policy_id"],
                "title": doc["title"],
                "category": doc["category"],
                "text": doc["text"],
                "relevance_score": round(score, 4)
            })
        return results

