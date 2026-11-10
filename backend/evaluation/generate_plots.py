"""
Evaluation Visualization & Figure Generator (Deliverable D4, D10)
Generates publication-quality charts saved to docs/figures/:
1. roc_curves.png: ROC-AUC curves comparing Baseline vs Improved perception models
2. confusion_matrix.png: 4-Class Decision Confusion Matrix (APPROVE, RETRY, MANUAL_REVIEW, REJECT)
3. risk_distribution.png: Calibrated Risk Score distributions across scenario categories
4. latency_histogram.png: End-to-end execution latency profile with p50 and p95 SLA markers
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

FIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "figures")
os.makedirs(FIGS_DIR, exist_ok=True)

# Set clean style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 0.8

def generate_all_plots():
    with open("backend/evaluation/benchmark_results.json") as f:
        bench_data = json.load(f)

    with open("backend/data/processed/bundles_manifest.json") as f:
        bundles = json.load(f)

    from backend.agent.controller import ReActAgentController
    from backend.perception.face_matcher import FaceMatcher
    from backend.perception.liveness_detector import LivenessDetector
    from backend.perception.audio_verifier import AudioVerifier

    controller = ReActAgentController()
    face_matcher = FaceMatcher()
    liveness_detector = LivenessDetector()
    audio_verifier = AudioVerifier()
    assets_dir = "backend/data/processed/assets"

    print("Generating evaluation figures for documentation...")

    # 1. ROC CURVES
    test_bundles = [b for b in bundles if b["split"] == "test"]
    face_gt, face_b_scores, face_i_scores = [], [], []
    live_gt, live_b_scores, live_i_scores = [], [], []
    rep_gt, rep_b_scores, rep_i_scores = [], [], []

    for b in test_bundles:
        selfie_p = os.path.join(assets_dir, b["verification_assets"]["selfie_image"])
        card_p = os.path.join(assets_dir, b["verification_assets"]["id_card_image"])
        audio_p = os.path.join(assets_dir, b["verification_assets"]["audio_challenge"])

        # Face
        face_gt.append(1 if b["verification_assets"]["face_match_expected"] else 0)
        face_b_scores.append(face_matcher.match(selfie_p, card_p, method="baseline")["similarity"])
        face_i_scores.append(face_matcher.match(selfie_p, card_p, method="improved")["similarity"])

        # Liveness
        live_gt.append(1 if not b["verification_assets"]["is_spoof_presentation"] else 0)
        live_b_scores.append(liveness_detector.detect(selfie_p, method="baseline")["liveness_score"])
        live_i_scores.append(liveness_detector.detect(selfie_p, method="improved")["liveness_score"])

        # Replay
        rep_gt.append(1 if b["verification_assets"]["is_replay_audio"] else 0)
        a_res = audio_verifier.decode_spoken_digits(audio_p, b["verification_assets"]["expected_challenge_digits"], bundle_metadata=b)
        rep_b_scores.append(a_res["replay_baseline"]["replay_score"])
        rep_i_scores.append(a_res["replay_analysis"]["replay_score"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Face ROC
    fpr_b, tpr_b, _ = roc_curve(face_gt, face_b_scores)
    fpr_i, tpr_i, _ = roc_curve(face_gt, face_i_scores)
    axes[0].plot(fpr_i, tpr_i, color='#2563eb', lw=2.5, label=f'Improved Deep ConvNet (AUC = {auc(fpr_i, tpr_i):.3f})')
    axes[0].plot(fpr_b, tpr_b, color='#94a3b8', lw=1.8, linestyle='--', label=f'Baseline Template/Hist (AUC = {auc(fpr_b, tpr_b):.3f})')
    axes[0].plot([0, 1], [0, 1], 'k:', alpha=0.3)
    axes[0].set_title('Face Biometric Verification', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Accept Rate (FAR)')
    axes[0].set_ylabel('True Accept Rate (1 - FRR)')
    axes[0].legend(loc='lower right', frameon=True)

    # Liveness ROC
    fpr_lb, tpr_lb, _ = roc_curve(live_gt, live_b_scores)
    fpr_li, tpr_li, _ = roc_curve(live_gt, live_i_scores)
    axes[1].plot(fpr_li, tpr_li, color='#059669', lw=2.5, label=f'Improved 2D FFT & Bezel (AUC = {auc(fpr_li, tpr_li):.3f})')
    axes[1].plot(fpr_lb, tpr_lb, color='#94a3b8', lw=1.8, linestyle='--', label=f'Baseline Chrominance (AUC = {auc(fpr_lb, tpr_lb):.3f})')
    axes[1].plot([0, 1], [0, 1], 'k:', alpha=0.3)
    axes[1].set_title('Facial Liveness (Anti-Spoofing)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(loc='lower right', frameon=True)

    # Replay ROC
    fpr_rb, tpr_rb, _ = roc_curve(rep_gt, rep_b_scores)
    fpr_ri, tpr_ri, _ = roc_curve(rep_gt, rep_i_scores)
    axes[2].plot(fpr_ri, tpr_ri, color='#d97706', lw=2.5, label=f'Improved SFM & Res (AUC = {auc(fpr_ri, tpr_ri):.3f})')
    axes[2].plot(fpr_rb, tpr_rb, color='#94a3b8', lw=1.8, linestyle='--', label=f'Baseline ZCR (AUC = {auc(fpr_rb, tpr_rb):.3f})')
    axes[2].plot([0, 1], [0, 1], 'k:', alpha=0.3)
    axes[2].set_title('Voice Acoustic Replay Detection', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('False Alarm Rate')
    axes[2].set_ylabel('Replay Detection Rate')
    axes[2].legend(loc='lower right', frameon=True)

    plt.tight_layout()
    fig_path1 = os.path.join(FIGS_DIR, "roc_curves.png")
    plt.savefig(fig_path1, dpi=200)
    plt.close()
    print(f"Saved: {fig_path1}")

    # 2. CONFUSION MATRIX & RISK DISTRIBUTIONS
    y_true = []
    y_pred = []
    category_risks = {"genuine_clean": [], "genuine_poor_capture": [], "spoof": [], "mismatch": []}
    latencies = []

    for b in bundles:
        r = controller.run_investigation(b, assets_dir)
        y_true.append(b["ground_truth_decision"])
        y_pred.append(r["final_decision"])
        category_risks[b["category"]].append(r["overall_risk_score"])
        latencies.append(r["total_latency_ms"])

    labels = ["APPROVE", "RETRY", "MANUAL_REVIEW", "REJECT"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(7, 6))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z}', ha='center', va='center',
                color='white' if z > np.max(cm)/2 else '#1e293b',
                fontweight='bold', fontsize=13)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='left')
    ax.set_yticklabels(labels)
    ax.set_xlabel('Agent Predicted Decision', fontweight='bold', labelpad=10)
    ax.set_ylabel('Ground-Truth Decision', fontweight='bold')
    ax.set_title('Multimodal eKYC Decision Confusion Matrix', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    fig_path2 = os.path.join(FIGS_DIR, "confusion_matrix.png")
    plt.savefig(fig_path2, dpi=200)
    plt.close()
    print(f"Saved: {fig_path2}")

    # 3. RISK DISTRIBUTION BOXPLOT
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_data = [category_risks["genuine_clean"], category_risks["genuine_poor_capture"],
                 category_risks["mismatch"], category_risks["spoof"]]
    bp = ax.boxplot(plot_data, patch_artist=True, tick_labels=['Genuine Clean', 'Poor Capture', 'Mismatch', 'Spoof Attack'])

    colors = ['#10b981', '#f59e0b', '#8b5cf6', '#f43f5e']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0.25, color='#10b981', linestyle='--', alpha=0.7, label='Auto-Approve Threshold (0.25)')
    ax.axhline(0.55, color='#f59e0b', linestyle='--', alpha=0.7, label='Manual Review Threshold (0.55)')
    ax.axhline(0.80, color='#f43f5e', linestyle='--', alpha=0.7, label='Quarantine/Reject Threshold (0.80)')
    ax.set_ylabel('Calibrated Risk Score (0.0 - 1.0)', fontweight='bold')
    ax.set_title('Calibrated Risk Score Distribution by Scenario Category', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    fig_path3 = os.path.join(FIGS_DIR, "risk_distribution.png")
    plt.savefig(fig_path3, dpi=200)
    plt.close()
    print(f"Saved: {fig_path3}")

    # 4. LATENCY PROFILE HISTOGRAM
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(latencies, bins=25, color='#3b82f6', edgecolor='#1d4ed8', alpha=0.75)
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    ax.axvline(p50, color='#10b981', lw=2, linestyle='-', label=f'Median (p50): {p50:.1f} ms')
    ax.axvline(p95, color='#f43f5e', lw=2, linestyle='--', label=f'95th Percentile (p95): {p95:.1f} ms')
    ax.set_xlabel('End-to-End Latency (ms)', fontweight='bold')
    ax.set_ylabel('Investigation Count')
    ax.set_title('End-to-End Autonomous Investigation Latency SLA Profile', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    fig_path4 = os.path.join(FIGS_DIR, "latency_histogram.png")
    plt.savefig(fig_path4, dpi=200)
    plt.close()
    print(f"Saved: {fig_path4}")

if __name__ == "__main__":
    generate_all_plots()

