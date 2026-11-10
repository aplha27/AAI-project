#!/usr/bin/env bash
set -e

echo "========================================================================="
echo "Starting Agentic Multimodal eKYC Verification Platform (HSR-AGAI-029)"
echo "========================================================================="

if [ ! -f "backend/data/processed/bundles_manifest.json" ]; then
    echo "[1/3] Generating 200 synthetic identity bundles..."
    python backend/data/generator.py
else
    echo "[1/3] Synthetic dataset found."
fi

echo "[2/3] Running automated test suite..."
python tests/test_full_suite.py

echo "[3/3] Launching FastAPI Backend & HITL Reviewer Console..."
echo "Console URL: http://localhost:8000"
echo "API Docs:    http://localhost:8000/docs"
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000

