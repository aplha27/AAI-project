# Dataset Card: Synthetic Multimodal eKYC Verification Benchmark

**Project ID:** HSR-AGAI-029 (Agentic Multimodal eKYC Verification, Risk Fusion and Escalation Agent)  
**Deliverable:** D2 – Documented Dataset & Preprocessing  
**Version:** 1.0.0 (Fixed Split Edition)  
**Date:** November 2026  

---

## 1. Overview & Purpose
In identity verification and banking fraud mitigation (modeled after industry references like FRS Labs), models must be evaluated against diverse multimodal artifacts without exposing private customer personally identifiable information (PII) or breaching regulatory privacy mandates.

This dataset provides **200 fully synthetic multimodal identity bundles** structured specifically for evaluating autonomous verification agents, perception models, and human-in-the-loop escalation systems.

Each bundle contains:
1. **Facial Selfie Image** (RGB, 320x360 px): Live applicant photo or presentation spoof.
2. **ID Document Image** (RGB, 600x380 px): Synthetic Republic National ID or Driving Licence containing photo portrait, security background, field text, and digital chip emulation.
3. **Spoken 4-Digit Challenge Audio** (WAV, 16kHz, 16-bit mono): Spoken digit sequence (e.g. "4 8 2 9") with acoustic variations, environmental noise, or phone speaker replay degradation.
4. **Structured Application Metadata**: Form input fields (name, DOB, ID number, address, device fingerprint, IP address, and submission velocity).

---

## 2. Dataset Partitioning & Splits
The dataset is split by identity into fixed, reproducible partitions:

| Split | Count | Proportion | Purpose |
|---|---|---|---|
| **Train** | 140 | 70% | Feature extraction, calibration parameter fitting, Q-learning policy simulation |
| **Validation** | 30 | 15% | Risk threshold tuning, stopping policy hyperparameter optimization |
| **Test** | 30 | 15% | Held-out benchmark evaluation (FAR/FRR, ROC-AUC, Agent Decision Accuracy) |
| **Total** | **200** | **100%** | Full Benchmark Suite |

---

## 3. Class & Scenario Distribution

| Category | Count | Expected Decision | Failure/Attack Vector Simulated |
|---|---|---|---|
| **Genuine, Clean** | 120 | `APPROVE` | Crisp capture, high face similarity (>0.85), genuine liveness, matching spoken challenge, unique device. |
| **Genuine, Poor Capture** | 30 | `RETRY` | Motion blur, document glare, low contrast, boundary occlusions. Valid identity requiring re-upload. |
| **Spoof / Fraud Attack** | 30 | `REJECT` (via HITL) | Screen presentation attack (moiré patterns, reflection), voice replay attack, compromised device ID cluster (`DEV-FRAUD-CLUSTER-X89`). |
| **Mismatch** | 20 | `MANUAL_REVIEW` | Discrepancy between application text and OCR text (e.g., misspelled name, mismatched DOB) or impostor face. |

---

## 4. Preprocessing & Feature Extraction Pipeline
1. **Vision Preprocessing**:
   - Resizing to standard dimensions (Selfie: 224x224, ID Card: 600x380).
   - Facial ROI detection and extraction using Haar Cascade / facial geometry bounding boxes.
   - Quality metrics: Laplacian variance blur score ($V_{lap} = \text{Var}(\nabla^2 I)$) and specular highlight glare score ($G_{glare}$).
2. **Acoustic Preprocessing**:
   - Normalization of audio amplitude to $-1.0 \dots +1.0$.
   - Short-Time Fourier Transform (STFT) spectrogram computation.
   - Spectral flatness and harmonic-to-noise ratio (HNR) extraction for replay-artefact detection.
3. **Text & Normalization**:
   - String canonicalization (lowercasing, punctuation stripping, whitespace trimming).
   - Normalized Levenshtein distance and token similarity ratio calculation.

---

## 5. Ethical Considerations & Privacy Compliance
- **Zero Real PII**: All names, dates, addresses, and ID numbers are algorithmically synthesized.
- **Biometric Safety**: Facial portraits are generated mathematically without scraping or republishing real biometric facial repositories.
- **Licensing**: Permissive MIT / Academic Open Source License.

