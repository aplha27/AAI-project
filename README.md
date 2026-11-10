# AAI-project: Agentic Multimodal eKYC Verification, Risk Fusion and Escalation Agent (HSR-AGAI-029)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark Accuracy](https://img.shields.io/badge/Accuracy-99.0%25-brightgreen.svg)]()
[![p95 Latency](https://img.shields.io/badge/p95%20Latency-26.3%20ms-cyan.svg)]()

> **Advanced AI / Agentic AI Capstone Project**  
> **Industry Reference:** FRS Labs (Identity verification and automated fraud prevention for banks)  
> **Specification:** Based on [ERP – Project Execution & Resource Plan](ERP%20–%20Project%20Execution%20&%20Resource%20Plan.pdf)

---

## 🌟 Executive Overview
This repository contains the complete production-grade implementation of **HSR-AGAI-029**: an autonomous multimodal eKYC agent that combines **Vision + Text + Audio perception**, evaluates risk via a **Calibrated Logistic Risk Fusion Engine**, grounds reasoning in **Regulatory Banking Runbooks (RBI / FATF)** through **Semantic Vector RAG**, and autonomously executes an inspectable **ReAct Agent Loop** (`Observe -> Reason -> Act -> Observe -> Escalate`).

---

## 🏗️ Architecture & Core Agent Loop

```
                     ┌──────────────────────────────────────────┐
                     │    Candidate Identity Application        │
                     │  (Selfie, ID Card, Audio, Form Data)     │
                     └────────────────────┬─────────────────────┘
                                          │
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │      Autonomous ReAct Controller         │
                     │  (Observe ──> Reason ──> Act ──> Escalate│
                     └──────┬─────────────┬─────────────┬───────┘
                            │             │             │
        ┌───────────────────┴──┐   ┌──────┴─────────┐  ┌┴─────────────────────┐
        │ Multimodal Perception│   │ Semantic RAG   │  │ Calibrated Risk      │
        │ - Face Matching      │   │ - KYC Runbooks │  │   Fusion Engine      │
        │ - OCR Sharpness/Glare│   │ - RBI/FATF Pol.│  │ - Logistic Sigmoid   │
        │ - Liveness FFT Anti-S│   │ - Fraud History│  │ - Compound Attacks   │
        │ - 4-Digit Spoken ASR │   └────────────────┘  └──────────┬───────────┘
        │ - Acoustic Replay SFM│                                  │
        └──────────────────────┘                                  ▼
                                                       ┌─────────────────────┐
                                                       │ Safety Guardrails   │
                                                       └──────────┬──────────┘
                                                                  │
               ┌──────────────────────┬───────────────────────────┴──────────────────────┐
               ▼                      ▼                                                  ▼
     ┌──────────────────┐   ┌──────────────────┐                               ┌──────────────────┐
     │     APPROVE      │   │      RETRY       │                               │ REJECT / MANUAL  │
     │ Straight-Through │   │ Image Recapture  │                               │ Escalated to     │
     │ (Risk < 0.25)    │   │ (Blur/Glare)     │                               │ HITL Reviewer    │
     └──────────────────┘   └──────────────────┘                               └────────┬─────────┘
                                                                                        │
                                                                                        ▼
                                                                               ┌──────────────────┐
                                                                               │ HITL Reviewer    │
                                                                               │ Console & Active │
                                                                               │ Learning Store   │
                                                                               └──────────────────┘
```

---

## 📦 Deliverables Traceability Matrix (ERP Scope)

| Deliverable | Description | Location in Repo | Status |
|---|---|---|---|
| **D1** | Single-command runnable repository | `run.bat`, `run.sh`, `docker-compose.yml` | **COMPLETED** |
| **D2** | Documented dataset & preprocessing | `backend/data/dataset_card.md`, `data_card.json` | **COMPLETED** |
| **D3** | Baseline vs Improved models | `backend/perception/`, `backend/fusion/` | **COMPLETED** |
| **D4** | Quantitative evaluation (FAR/FRR/AUC) | `docs/EVALUATION_REPORT.md`, `benchmark_results.json` | **COMPLETED** |
| **D5** | Architecture & state-action diagrams | `docs/ARCHITECTURE_DIAGRAM.md` | **COMPLETED** |
| **D6** | FastAPI backend service (p95 SLA) | `backend/main.py` | **COMPLETED** |
| **D7** | HITL Reviewer Console / Dashboard | `frontend/index.html` | **COMPLETED** |
| **D8** | Replayable tool-call action trace | In every case response (`action_trace` key) | **COMPLETED** |
| **D9** | Deterministic guardrail test suite | `tests/test_full_suite.py` (13/13 passing) | **COMPLETED** |
| **D10** | Benchmark & failure analysis report | `docs/EVALUATION_REPORT.md` | **COMPLETED** |
| **D11** | Documentation & deployment guide | `README.md`, `Dockerfile` | **COMPLETED** |
| **D12** | Comprehensive capstone report | `docs/FINAL_REPORT.md` | **COMPLETED** |
| **D13** | Live demonstration video/demo script | `docs/DEMO_SCRIPT.md` | **COMPLETED** |

---

## 🚀 Quickstart Guide

### Option 1: Native Local Run (Windows / Linux / macOS)

**Prerequisites:** Python 3.10+ installed.

#### On Windows:
```cmd
.\run.bat
```

#### On Linux / macOS:
```bash
chmod +x run.sh
./run.sh
```

The script will automatically:
1. Verify the 200 synthetic identity bundles.
2. Run the automated test suite (unit tests, guardrails, API lifecycle).
3. Start the FastAPI server on **`http://127.0.0.1:8000`**.

---

### Option 2: Docker / Containerized Run

```bash
# Build and run with Docker Compose
docker-compose up --build
```
Navigate to: **`http://localhost:8000`**

---

## 🖥️ Interactive Reviewer Console & Demo Walkthrough

1. Open your browser at **`http://127.0.0.1:8000`**.
2. **Real-time Demo Launcher:** Use the top scenario buttons to trigger and observe:
   - **`✓ Clean Genuine (Approve)`**: Fast 7-hop straight-through verification ($R = 0.088$).
   - **`📷 Blurry Document (Retry)`**: Early-stopping at Step 2 issuing recapture instructions ($V_{lap} < 500$).
   - **`⚠️ Screen & Voice Spoof (Reject)`**: 2D FFT moiré detection + speaker replay detection ($R = 0.990$) quarantining to supervisor.
   - **`🔍 Impostor / Name Mismatch`**: Biometric skin reflectance & demographic mismatch triggering `MANUAL_REVIEW`.
3. **Supervisor Review Action Panel:** Reviewers can inspect side-by-side media, listen to 16kHz audio challenges, audit ReAct reasoning hops, and submit override decisions that log to the continuous learning store.

---

## 📊 Benchmark & Quantitative Evaluation Summary

From the held-out evaluation across all 200 identity bundles (`python backend/evaluation/evaluate.py`):

| Metric | Result | Benchmark Target | Verdict |
|---|---|---|---|
| **Overall Decision Accuracy** | **99.00%** | > 90.0% | **OUTSTANDING** |
| **False Acceptance Rate (FAR)** | **0.00%** | < 1.0% | **ZERO LEAKAGE** |
| **False Rejection Rate (FRR)** | **0.00%** | < 5.0% | **MINIMAL FRICTION** |
| **Facial Match ROC-AUC** | **0.9950** | > 0.90 | **STATE-OF-THE-ART** |
| **Liveness Anti-Spoof ROC-AUC** | **0.9920** | > 0.85 | **ROBUST** |
| **Acoustic Replay ROC-AUC** | **0.9880** | > 0.85 | **ROBUST** |
| **Mean End-to-End Latency** | **21.34 ms** | < 2000 ms | **CPU OPTIMIZED** |
| **95th Percentile Latency (p95)** | **26.31 ms** | < 3000 ms | **SLA MET** |

---

## 🛠️ Repository Directory Structure

```
├── backend/
│   ├── agent/
│   │   ├── controller.py       # ReAct Autonomous Controller & Investigation Loop
│   │   ├── case_memory.py      # Working Memory & Fraud Syndicate Store
│   │   └── rag/
│   │       └── policy_store.py # Regulatory KYC Runbooks & Vector Store (RAG)
│   ├── perception/
│   │   ├── face_matcher.py     # Deep ConvNet + Skin Reflectance Biometrics
│   │   ├── ocr_extractor.py    # Document Sharpness/Glare & Fuzzy Field Matching
│   │   ├── liveness_detector.py# 2D FFT Moiré & Physical Bezel Anti-Spoof
│   │   ├── audio_verifier.py   # Spoken 4-Digit ASR & Transducer Replay SFM
│   │   └── metadata_analyzer.py# Device Fingerprint & Syndicate Velocity Audit
│   ├── fusion/
│   │   └── risk_fusion.py      # Calibrated Logistic Non-Linear Risk Fusion Engine
│   ├── data/
│   │   ├── generator.py        # 200 Synthetic Bundle Generator
│   │   └── dataset_card.md     # Deliverable D2 Data Card
│   ├── evaluation/
│   │   ├── evaluate.py         # Full Quantitative Benchmark Runner (D4, D10)
│   │   └── benchmark_results.json # Serialized Evaluation Metrics
│   └── main.py                 # FastAPI Application & REST API
├── frontend/
│   └── index.html              # Modern HITL Reviewer Console Dashboard
├── docs/
│   ├── FINAL_REPORT.md         # Deliverable D12 Comprehensive Technical Report
│   ├── EVALUATION_REPORT.md    # Deliverable D4 & D10 Benchmark Report
│   ├── ARCHITECTURE_DIAGRAM.md # Deliverable D5 State-Action & Architecture Diagrams
│   ├── DECISION_LOG.md         # Architecture Decision Records (ADRs)
│   └── DEMO_SCRIPT.md          # Deliverable D13 Live Presentation Script
├── tests/
│   ├── test_agent_smoke.py     # Smoke Test across 4 Core Scenarios
│   └── test_full_suite.py      # Deliverable D9 Unit, Guardrail & API Test Suite
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
├── requirements.txt            # Python dependencies
├── run.bat                     # Windows one-click startup
└── run.sh                      # Unix/macOS one-click startup
```

---

## 📡 REST API Reference

- `POST /api/kyc/verify`: Autonomous investigation endpoint (returns decision, risk score, and full ReAct trace).
- `GET /api/kyc/cases`: Lists all verified cases in working memory with filter support.
- `GET /api/kyc/cases/{case_id}`: Retrieves complete forensic breakdown and replayable action trace.
- `GET /api/kyc/reviewer-queue`: Fetches pending cases escalated for human supervisor oversight.
- `POST /api/kyc/cases/{case_id}/review`: Logs supervisor verdict and feedback to the active learning pipeline.
- `GET /api/metrics`: Live SLA tracking, p95 latency, approval and escalation distribution.
- `GET /api/health`: Service health check and loaded model diagnostics.
- `GET /`: Interactive Human-In-The-Loop Reviewer Console.

---

## 👥 Contributors & Academic Acknowledgments
- **Project Title:** HSR-AGAI-029: Agentic Multimodal eKYC Verification, Risk Fusion and Escalation Agent
- **Course:** Advanced AI / Agentic AI
- **Reference Sponsor:** Modeled after FRS Labs identity verification systems for banking.

