"""
Smoke Test for ReAct Agent Controller across Core Scenarios
Tests:
1. Genuine Clean (Expect: APPROVE)
2. Poor Capture Blur/Glare (Expect: RETRY)
3. Spoof Presentation Attack (Expect: REJECT via HITL)
4. Demographic / Biometric Mismatch (Expect: MANUAL_REVIEW)
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from backend.agent.controller import ReActAgentController

def main():
    with open("backend/data/processed/bundles_manifest.json") as f:
        bundles = json.load(f)

    controller = ReActAgentController()

    # Find one of each category
    samples = {}
    for b in bundles:
        cat = b["category"]
        if cat not in samples:
            samples[cat] = b
        if len(samples) == 4:
            break

    print("\n--- RUNNING RE-ACT AGENT CONTROLLER SMOKE SUITE ---")
    all_passed = True
    for cat, b in samples.items():
        res = controller.run_investigation(b, "backend/data/processed/assets")
        match = res["decision_matches_ground_truth"]
        status = "PASSED" if match else "MISMATCH"
        if not match:
            all_passed = False
        print(f"[{status}] {b['bundle_id']} | Category: {cat:<22} | Decision: {res['final_decision']:<14} | GT: {b['ground_truth_decision']:<14} | Risk: {res['overall_risk_score']:<5} | Steps: {res['total_steps']}")

    print(f"\nSmoke Test Result: {'ALL SCENARIOS VERIFIED' if all_passed else 'SOME DISCREPANCIES DETECTED'}\n")

if __name__ == "__main__":
    main()
