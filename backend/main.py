"""
FastAPI Backend Application for Agentic Multimodal eKYC Verification & Escalation (HSR-AGAI-029)
Provides:
- REST API for autonomous verification and ReAct trace inspection
- Human-in-the-loop (HITL) reviewer console endpoints
- Metrics, SLA tracking (p95 latency), and health monitoring
- Static asset delivery for biometric media and frontend dashboard
"""

import os
import sys
import time
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

from fastapi import FastAPI, HTTPException, Query, Body, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from backend.agent.controller import ReActAgentController
from backend.agent.case_memory import CaseHistoryMemory

# Initialize paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "backend", "data", "processed", "assets")
MANIFEST_PATH = os.path.join(BASE_DIR, "backend", "data", "processed", "bundles_manifest.json")
BENCHMARK_PATH = os.path.join(BASE_DIR, "backend", "evaluation", "benchmark_results.json")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
FIGURES_DIR = os.path.join(BASE_DIR, "docs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

app = FastAPI(
    title="Agentic Multimodal eKYC Verification Platform",
    description="FRS Labs reference autonomous risk fusion and escalation agent (HSR-AGAI-029)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory persistence stores
memory_store = CaseHistoryMemory()
agent_controller = ReActAgentController(memory_store=memory_store)

# Load manifest and benchmark data
ALL_BUNDLES: List[Dict[str, Any]] = []
BUNDLE_MAP: Dict[str, Dict[str, Any]] = {}
CASES_STORE: Dict[str, Dict[str, Any]] = {}
REVIEWER_FEEDBACK_LOG: List[Dict[str, Any]] = []

if os.path.exists(MANIFEST_PATH):
    with open(MANIFEST_PATH, "r") as f:
        ALL_BUNDLES = json.load(f)
        BUNDLE_MAP = {b["bundle_id"]: b for b in ALL_BUNDLES}

# Models
class ReviewSubmission(BaseModel):
    reviewer_id: str = Field(..., example="REV-ANALYST-07")
    verdict: str = Field(..., example="APPROVE", pattern="^(APPROVE|REJECT|RETRY|FLAG_FRAUD)$")
    notes: str = Field(..., example="Biometric identity corroborated upon secondary visual inspection.")
    override_reason: Optional[str] = None

class VerificationRequest(BaseModel):
    bundle_id: Optional[str] = None
    custom_bundle: Optional[Dict[str, Any]] = None

# Pre-populate sample cases for immediate dashboard inspection
def init_sample_cases():
    if not ALL_BUNDLES:
        return
    # Process initial set of 20 diverse cases for console exploration
    for b in ALL_BUNDLES[:25]:
        res = agent_controller.run_investigation(b, ASSETS_DIR)
        CASES_STORE[res["case_id"]] = res

init_sample_cases()

# Mount static asset files
os.makedirs(ASSETS_DIR, exist_ok=True)
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount("/figures", StaticFiles(directory=FIGURES_DIR), name="figures")

from backend.agent.rl_policy import QLearningDecisionPolicy
from backend.evaluation.retraining_pipeline import RetrainingPipeline

rl_policy = QLearningDecisionPolicy()
retraining_pipeline = RetrainingPipeline()

if os.path.exists(os.path.join(FRONTEND_DIR, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_DIR, "static")), name="static")


# ---------------- API ENDPOINTS ----------------

@app.get("/api/rl-policy")
def get_rl_policy_status():
    return {
        "status": "ACTIVE",
        "policy_type": "hybrid_rules_plus_qlearning",
        "epsilon": rl_policy.epsilon,
        "converged_states": len(rl_policy.q_table),
        "actions": ["APPROVE", "RETRY", "MANUAL_REVIEW", "REJECT"]
    }

@app.post("/api/retrain")
def trigger_retraining():
    feedback = retraining_pipeline.simulate_reviewer_feedback(MANIFEST_PATH, count=40)
    current_w = agent_controller.risk_fusion.weights
    checkpoint = retraining_pipeline.fine_tune_fusion_weights(feedback, current_w)
    agent_controller.risk_fusion.weights = checkpoint["fine_tuned_weights"]
    return {
        "status": "RETRAINED",
        "version": checkpoint["version"],
        "updated_weights": checkpoint["fine_tuned_weights"],
        "loss": checkpoint["final_cross_entropy_loss"]
    }

@app.get("/api/health")
def health_check():
    return {
        "status": "HEALTHY",
        "service": "Agentic Multimodal eKYC Verification Agent",
        "project_id": "HSR-AGAI-029",
        "version": "1.0.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models_loaded": {
            "face_matcher": "LightweightFaceNet (PyTorch 128-d Embeddings)",
            "ocr_extractor": "Quality-Aware Laplacian & Specular Analyzer",
            "liveness_detector": "Multi-Cue 2D FFT Moiré & Bezel Detector",
            "audio_verifier": "Formant ASR & Replay SFM Detector",
            "risk_fusion": "Calibrated Logistic Non-Linear Interaction Model"
        },
        "total_bundles_available": len(ALL_BUNDLES),
        "total_cases_processed": len(CASES_STORE)
    }

@app.get("/api/kyc/bundles")
def list_available_bundles(category: Optional[str] = None, limit: int = 50):
    """
    Returns available synthetic test bundles from the 200 identity catalog.
    """
    results = ALL_BUNDLES
    if category:
        results = [b for b in results if b.get("category") == category]
    return results[:limit]

@app.post("/api/kyc/verify")
def verify_identity(req: VerificationRequest):
    """
    Main Autonomous Verification Endpoint:
    Executes the multi-hop ReAct agent loop, returning decisions, risk fusion scores,
    and the full replayable step-by-step tool trace.
    """
    if req.bundle_id:
        if req.bundle_id not in BUNDLE_MAP:
            raise HTTPException(status_code=404, detail=f"Bundle ID {req.bundle_id} not found in catalog.")
        target_bundle = BUNDLE_MAP[req.bundle_id]
    elif req.custom_bundle:
        target_bundle = req.custom_bundle
    else:
        # Default to first available bundle
        target_bundle = ALL_BUNDLES[0] if ALL_BUNDLES else {}

    result = agent_controller.run_investigation(target_bundle, ASSETS_DIR)
    CASES_STORE[result["case_id"]] = result
    return result

@app.get("/api/kyc/cases")
def list_cases(status: Optional[str] = None, category: Optional[str] = None):
    """
    Retrieves all verified cases stored in working memory.
    """
    cases = list(CASES_STORE.values())
    if status:
        if status.upper() == "ESCALATED":
            cases = [c for c in cases if c.get("escalated_to_hitl")]
        else:
            cases = [c for c in cases if c.get("final_decision") == status.upper()]
    if category:
        cases = [c for c in cases if c.get("bundle_category") == category]
    return cases

@app.get("/api/kyc/cases/{case_id}")
def get_case_detail(case_id: str):
    """
    Retrieves complete case details with deep ReAct trace timeline.
    """
    if case_id in CASES_STORE:
        return CASES_STORE[case_id]
    # Check if in bundle catalog and run on demand
    if case_id in BUNDLE_MAP:
        res = agent_controller.run_investigation(BUNDLE_MAP[case_id], ASSETS_DIR)
        CASES_STORE[case_id] = res
        return res
    raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

@app.get("/api/kyc/reviewer-queue")
def get_reviewer_queue():
    """
    Fetches pending cases requiring Human-In-The-Loop review.
    """
    queue = [
        c for c in CASES_STORE.values()
        if c.get("escalated_to_hitl") and not c.get("reviewed_by_human")
    ]
    return queue

@app.post("/api/kyc/cases/{case_id}/review")
def submit_human_review(case_id: str, submission: ReviewSubmission):
    """
    HITL Reviewer Endpoint:
    Logs human supervisor decisions, updates case verdict, and stores ground-truth active learning feedback.
    """
    if case_id not in CASES_STORE:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found.")

    case = CASES_STORE[case_id]
    case["reviewed_by_human"] = True
    case["reviewer_verdict"] = submission.verdict
    case["reviewer_id"] = submission.reviewer_id
    case["reviewer_notes"] = submission.notes
    case["reviewer_override"] = (submission.verdict != case["final_decision"])
    case["review_timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Log to active learning feedback store
    feedback_entry = {
        "case_id": case_id,
        "agent_decision": case["final_decision"],
        "human_verdict": submission.verdict,
        "is_override": case["reviewer_override"],
        "notes": submission.notes,
        "perception_features": case.get("perception_summary"),
        "timestamp": case["review_timestamp"]
    }
    REVIEWER_FEEDBACK_LOG.append(feedback_entry)

    return {
        "status": "REVIEW_RECORDED",
        "case_id": case_id,
        "new_status": submission.verdict,
        "is_override": case["reviewer_override"],
        "feedback_queue_size": len(REVIEWER_FEEDBACK_LOG)
    }

@app.get("/api/metrics")
def get_system_metrics():
    """
    Computes real-time system metrics, SLA adherence, and decision distribution.
    """
    total = len(CASES_STORE)
    if total == 0:
        return {"total_cases": 0}

    approved = sum(1 for c in CASES_STORE.values() if c.get("final_decision") == "APPROVE")
    rejected = sum(1 for c in CASES_STORE.values() if c.get("final_decision") == "REJECT")
    retry = sum(1 for c in CASES_STORE.values() if c.get("final_decision") == "RETRY")
    manual = sum(1 for c in CASES_STORE.values() if c.get("final_decision") == "MANUAL_REVIEW")
    escalated = sum(1 for c in CASES_STORE.values() if c.get("escalated_to_hitl"))
    reviewed = sum(1 for c in CASES_STORE.values() if c.get("reviewed_by_human"))

    latencies = [c.get("total_latency_ms", 40.0) for c in CASES_STORE.values()]
    mean_lat = round(sum(latencies) / float(total), 2)
    p95_lat = round(float(sorted(latencies)[int(0.95 * total)]), 2) if total >= 20 else mean_lat

    # Load benchmark summary if present
    benchmark_data = None
    if os.path.exists(BENCHMARK_PATH):
        with open(BENCHMARK_PATH) as f:
            benchmark_data = json.load(f)

    return {
        "total_cases_processed": total,
        "decisions": {
            "approved": approved,
            "rejected": rejected,
            "retry": retry,
            "manual_review": manual
        },
        "rates": {
            "approval_rate": round(approved / float(total), 4),
            "rejection_rate": round(rejected / float(total), 4),
            "retry_rate": round(retry / float(total), 4),
            "escalation_rate": round(escalated / float(total), 4),
            "human_resolution_rate": round(reviewed / float(max(1, escalated)), 4)
        },
        "latency_profile": {
            "mean_ms": mean_lat,
            "p95_ms": p95_lat
        },
        "offline_benchmark": benchmark_data.get("agentic_performance") if benchmark_data else None,
        "perception_roc_auc": benchmark_data.get("perception_comparison") if benchmark_data else None
    }

# ---------------- FRONTEND DASHBOARD ----------------

@app.get("/", response_class=HTMLResponse)
@app.get("/console", response_class=HTMLResponse)
def serve_dashboard():
    dashboard_html_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(dashboard_html_path):
        with open(dashboard_html_path, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>Agentic Multimodal eKYC Verification Platform</h1><p>Frontend dashboard loading...</p>"

if __name__ == "__main__":
    import uvicorn
    print("Starting eKYC Verification API on port 8000...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
