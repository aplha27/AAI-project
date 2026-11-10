# Agent Architecture & State-Action Diagrams (Deliverable D5)

**Project ID:** HSR-AGAI-029 (Agentic Multimodal eKYC Verification, Risk Fusion and Escalation Agent)  
**Deliverable:** D5 – Agent Architecture and State-Action Diagram  

---

## 1. System High-Level Architecture Diagram

```mermaid
flowchart TD
    subgraph ClientLayer["Client & Ingestion Layer"]
        User["Applicant (Mobile / Web Client)"]
        Reviewer["Compliance Officer (HITL Console)"]
    end

    subgraph PerceptionSuite["Multimodal Perception Suite"]
        P1["Face Matcher (Deep ConvNet Embedding)"]
        P2["OCR & Document Quality (Sharpness & Glare)"]
        P3["Facial Liveness (2D FFT Moiré & Bezel)"]
        P4["Audio Verifier (ASR Formant & Replay SFM)"]
        P5["Metadata & Syndicate Intelligence"]
    end

    subgraph AgentCore["Autonomous Agent Core"]
        ReAct["ReAct Controller (Observe-Reason-Act)"]
        RiskFusion["Calibrated Logistic Risk Fusion Engine"]
        PolicyRAG["KYC Regulatory Policy Corpus & Vector Store (RAG)"]
        CaseMemory["Case Working Memory & Fraud Syndicate History"]
    end

    subgraph DecisionLayer["Decision & Escalation Guardrails"]
        G1{"Deterministic Guardrails"}
        Approve["APPROVE (Straight-Through)"]
        Retry["RETRY (Image Re-capture)"]
        Review["MANUAL_REVIEW (Investigator Sign-off)"]
        Reject["REJECT (Quarantined via HITL)"]
    end

    User -->|"Submit Biometrics (Selfie, ID Card, Audio)"| ReAct
    ReAct <-->|"Tool Calls & Observations"| PerceptionSuite
    ReAct <-->|"Semantic Rule Retrieval"| PolicyRAG
    ReAct <-->|"Fraud History Lookup"| CaseMemory
    ReAct -->|"Synthesize State"| RiskFusion
    RiskFusion --> G1

    G1 -->|"Clean Pass (Risk < 0.25)"| Approve
    G1 -->|"Blur / Glare (Attempts < 3)"| Retry
    G1 -->|"Discrepancy (Face/Name)"| Review
    G1 -->|"Presentation Attack / Fraud Cluster"| Reject

    Review -->|"Escalate Case"| Reviewer
    Reject -->|"Escalate Case"| Reviewer
    Reviewer -->|"Ground-Truth Feedback & Retraining Store"| CaseMemory
```

---

## 2. ReAct Agent State-Action Finite State Machine (FSM)

```mermaid
stateDiagram-v2
    [*] --> STATE_INGESTION: Candidate Submits Identity Bundle

    STATE_INGESTION --> STATE_DOC_QUALITY: TOOL_OCR_EXTRACT_AND_CHECK
    
    STATE_DOC_QUALITY --> STATE_DECISION_RETRY: Guardrail 1 (Blur < 500 or Glare Washout)
    STATE_DOC_QUALITY --> STATE_BIOMETRIC_EVAL: Pass Optical Quality Gate

    STATE_BIOMETRIC_EVAL --> STATE_LIVENESS_CHECK: TOOL_VERIFY_FACE_MATCH
    
    STATE_LIVENESS_CHECK --> STATE_AUDIO_CHALLENGE: TOOL_VERIFY_LIVENESS
    
    STATE_AUDIO_CHALLENGE --> STATE_METADATA_AUDIT: TOOL_VERIFY_SPOKEN_CHALLENGE
    
    STATE_METADATA_AUDIT --> STATE_RISK_FUSION: TOOL_ANALYZE_METADATA_AND_DEVICE
    
    STATE_RISK_FUSION --> STATE_POLICY_GROUNDING: TOOL_FUSE_MULTIMODAL_RISK

    STATE_POLICY_GROUNDING --> STATE_GUARDRAIL_EVAL: TOOL_RAG_QUERY_KYC_POLICY

    state STATE_GUARDRAIL_EVAL {
        [*] --> CheckSpoof
        CheckSpoof --> TriggerReject: Liveness < 0.65 OR Replay OR Fraud Cluster
        CheckSpoof --> CheckDiscrepancy: Liveness OK
        CheckDiscrepancy --> TriggerManual: Face < 0.70 OR Name/DOB Mismatch
        CheckDiscrepancy --> TriggerApprove: Risk < 0.25 & All Checks Pass
    }

    TriggerReject --> STATE_DECISION_REJECT: Mandate HITL Escalation
    TriggerManual --> STATE_DECISION_MANUAL: Mandate HITL Escalation
    TriggerApprove --> STATE_DECISION_APPROVE: Straight-Through Execution

    STATE_DECISION_RETRY --> [*]: Return Recapture Instructions
    STATE_DECISION_APPROVE --> [*]: Issue Verified Status
    STATE_DECISION_MANUAL --> STATE_HITL_QUEUE: Forward Case & Trace
    STATE_DECISION_REJECT --> STATE_HITL_QUEUE: Forward Case & Trace

    STATE_HITL_QUEUE --> STATE_FEEDBACK_RECORD: Human Supervisor Signs Off
    STATE_FEEDBACK_RECORD --> [*]: Log to Active Learning Store
```

---

## 3. Labelled Tool Registry

| Tool Identifier | Input Parameters | Output Artifacts | Governing Policy |
|---|---|---|---|
| `TOOL_OCR_EXTRACT_AND_CHECK` | `card_image_path`, `form_data` | OCR fields, blur metric, glare ratio, text consistency score | `POL-001`, `POL-004` |
| `TOOL_VERIFY_FACE_MATCH` | `selfie_image_path`, `card_image_path` | Cosine similarity, skin reflectance distance, match verdict | `POL-003` |
| `TOOL_VERIFY_LIVENESS` | `selfie_image_path` | 2D FFT moiré peak, device bezel detection, calibrated liveness score | `POL-002` |
| `TOOL_VERIFY_SPOKEN_CHALLENGE` | `audio_path`, `expected_digits` | Digit Error Rate (DER), spectral flatness, acoustic replay probability | `POL-005` |
| `TOOL_ANALYZE_METADATA_AND_DEVICE` | `applicant_metadata` | Known fraud cluster collision, hardware velocity count, metadata risk | `POL-006` |
| `TOOL_RAG_QUERY_KYC_POLICY` | `semantic_query_string` | Top-k relevant KYC runbook rules with authoritative citations | All (`POL-001` - `POL-006`) |
| `TOOL_QUERY_CASE_MEMORY` | `device_id`, `id_number` | Working memory history, previous attempts, repeat rejections | Fraud Detection Runbook |
| `TOOL_FUSE_MULTIMODAL_RISK` | `perception_bundle` | Calibrated overall risk score ($R \in [0.0, 1.0]$), risk tier, breakdown matrix | Banking Risk Standard |
| `TOOL_EMIT_DECISION` | `decision`, `escalate`, `rationale` | Final terminal action (`APPROVE`, `RETRY`, `MANUAL_REVIEW`, `REJECT`) | Executive Policy |

