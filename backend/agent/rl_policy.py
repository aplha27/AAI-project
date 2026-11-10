"""
Reinforcement Learning & Contextual Bandit Decision Policy (Week 8 of ERP)
Implements:
1. State Discretization / Feature Binning
2. Contextual Bandit / Q-Learning Decision Policy over Action Space:
   - Action 0: APPROVE
   - Action 1: RETRY
   - Action 2: MANUAL_REVIEW
   - Action 3: REJECT
3. Banking Utility Reward Model (penalizes fraud false acceptance heavily: -15.0)
4. Rules-Plus-Bandit Hybrid Guardrails
5. Monte Carlo Simulation Runner for Q-learning Convergence & Evaluation
"""

import math
import random
import json
import numpy as np
from typing import Dict, Any, Tuple, List

ACTIONS = ["APPROVE", "RETRY", "MANUAL_REVIEW", "REJECT"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

# Banking Utility / Risk Reward Matrix
# R(action, ground_truth)
REWARD_MATRIX = {
    # If Ground Truth is APPROVE (Genuine Clean)
    "APPROVE": {
        "APPROVE": 1.0,        # Straight-through convenience
        "RETRY": -0.5,          # Unnecessary friction
        "MANUAL_REVIEW": -0.4,  # Unnecessary human operational cost
        "REJECT": -5.0          # Unfair denial of service
    },
    # If Ground Truth is RETRY (Poor Capture / Blur / Glare)
    "RETRY": {
        "APPROVE": -3.0,        # Approving unverified illegible doc
        "RETRY": 0.9,           # Appropriate guidance to user
        "MANUAL_REVIEW": -0.3,  # Human review of unreadable image wastes time
        "REJECT": -2.0          # Unfair to reject recoverable capture
    },
    # If Ground Truth is REJECT (Spoof / Replay / Fraud Cluster)
    "REJECT": {
        "APPROVE": -15.0,       # Catastrophic fraud loss (False Acceptance)
        "RETRY": -4.0,          # Allowing attacker to retry presentation attack
        "MANUAL_REVIEW": 0.5,   # Human investigator catches it
        "REJECT": 1.5           # Clean fraud mitigation
    },
    # If Ground Truth is MANUAL_REVIEW (Demographic or Face Mismatch)
    "MANUAL_REVIEW": {
        "APPROVE": -10.0,       # Potential identity theft
        "RETRY": -1.0,          # Won't fix intrinsic mismatch
        "MANUAL_REVIEW": 1.0,   # Correct escalation for corroboration
        "REJECT": -1.5          # Premature rejection of typo
    }
}

class QLearningDecisionPolicy:
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, epsilon: float = 0.15):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[str, np.ndarray] = {}
        self.training_history: List[Dict[str, Any]] = []

    def get_state_key(self, features: Dict[str, float]) -> str:
        """
        Discretizes continuous multimodal perception scores into a compact state hash:
        - face_bin: Low (<0.65), Med (0.65-0.75), High (>=0.75)
        - live_bin: Spoof (<0.65), Live (>=0.65)
        - doc_bin: Degraded (<0.60), Crisp (>=0.60)
        - text_bin: Mismatch (<0.85), Match (>=0.85)
        - audio_bin: Replay/DER (<0.50), Clean (>=0.50)
        - meta_bin: HighRisk (>=0.50), Clean (<0.50)
        """
        f_face = "H" if features.get("face_sim", 0.9) >= 0.75 else ("M" if features.get("face_sim", 0.9) >= 0.65 else "L")
        f_live = "LIV" if features.get("liveness", 0.9) >= 0.65 else "SPF"
        f_doc = "CRS" if features.get("doc_quality", 0.9) >= 0.60 else "BLR"
        f_text = "MTC" if features.get("text_consistency", 1.0) >= 0.85 else "MIS"
        f_audio = "CLN" if features.get("audio_clean", 1.0) >= 0.50 else "REP"
        f_meta = "SAF" if features.get("meta_risk", 0.05) < 0.50 else "FRD"

        return f"{f_face}_{f_live}_{f_doc}_{f_text}_{f_audio}_{f_meta}"

    def get_q_values(self, state: str) -> np.ndarray:
        if state not in self.q_table:
            # Initialize with heuristic prior
            # Actions: [APPROVE, RETRY, MANUAL_REVIEW, REJECT]
            self.q_table[state] = np.zeros(len(ACTIONS), dtype=np.float32)
            # Prior initialization: If degraded, bump RETRY; if spoof, bump REJECT
            if "BLR" in state:
                self.q_table[state][ACTION_TO_IDX["RETRY"]] += 0.5
            if "SPF" in state or "FRD" in state or "REP" in state:
                self.q_table[state][ACTION_TO_IDX["REJECT"]] += 0.8
            if "MIS" in state or "L" in state:
                self.q_table[state][ACTION_TO_IDX["MANUAL_REVIEW"]] += 0.6
            if "H" in state and "LIV" in state and "CRS" in state and "MTC" in state and "CLN" in state and "SAF" in state:
                self.q_table[state][ACTION_TO_IDX["APPROVE"]] += 1.0
        return self.q_table[state]

    def select_action(self, state: str, explore: bool = True) -> int:
        """
        Epsilon-greedy action selection.
        """
        if explore and random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS) - 1)
        q_vals = self.get_q_values(state)
        # Random tie-breaking
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(random.choice(best_actions))

    def update(self, state: str, action: int, reward: float, next_state: str = None):
        """
        Q-learning Bellman update.
        """
        current_q = self.get_q_values(state)[action]
        if next_state:
            max_next_q = np.max(self.get_q_values(next_state))
        else:
            max_next_q = 0.0  # Terminal state

        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q

    def train_on_manifest(self, bundles: List[Dict[str, Any]], episodes: int = 1500) -> Dict[str, Any]:
        """
        Runs episodic simulation across training bundles to converge Q-values.
        """
        train_bundles = [b for b in bundles if b.get("split") == "train"] or bundles
        rewards_per_epoch = []

        for ep in range(episodes):
            total_epoch_reward = 0.0
            random.shuffle(train_bundles)

            for b in train_bundles:
                # Build feature vector
                gt_dec = b.get("ground_truth_decision", "APPROVE")
                assets = b.get("verification_assets", {})
                doc_data = b.get("document_data", {})

                face_sim = 0.90 if assets.get("face_match_expected", True) else 0.45
                liveness = 0.20 if assets.get("is_spoof_presentation", False) else 0.95
                doc_qual = 0.30 if (doc_data.get("blur_level", 0.0) > 0 or doc_data.get("glare_present", False)) else 0.95
                text_cons = 0.40 if b.get("category") == "mismatch" else 1.0
                audio_clean = 0.10 if assets.get("is_replay_audio", False) else 0.95
                meta_risk = 0.85 if "FRAUD" in b.get("applicant_data", {}).get("device_id", "") else 0.05

                features = {
                    "face_sim": face_sim,
                    "liveness": liveness,
                    "doc_quality": doc_qual,
                    "text_consistency": text_cons,
                    "audio_clean": audio_clean,
                    "meta_risk": meta_risk
                }

                state = self.get_state_key(features)
                action_idx = self.select_action(state, explore=True)
                chosen_action = IDX_TO_ACTION[action_idx]

                reward = REWARD_MATRIX.get(gt_dec, {}).get(chosen_action, 0.0)
                self.update(state, action_idx, reward, next_state=None)
                total_epoch_reward += reward

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.995)
            if ep % 100 == 0 or ep == episodes - 1:
                avg_reward = total_epoch_reward / len(train_bundles)
                rewards_per_epoch.append({"episode": ep, "avg_reward": round(avg_reward, 3)})

        return {
            "converged_states": len(self.q_table),
            "final_epsilon": round(self.epsilon, 4),
            "training_curve": rewards_per_epoch
        }

    def predict_hybrid(self, features: Dict[str, float]) -> Dict[str, Any]:
        """
        Hybrid Rules-Plus-Bandit Policy:
        Enforces hard regulatory guardrails as deterministic constraints;
        utilizes learned Q-values for boundary / borderline cases.
        """
        state = self.get_state_key(features)
        q_vals = self.get_q_values(state).copy()

        # Hard Guardrail 1: Presentation spoof / fraud cluster must NEVER be approved
        if features.get("liveness", 1.0) < 0.65 or features.get("meta_risk", 0.0) > 0.50 or features.get("audio_clean", 1.0) < 0.50:
            q_vals[ACTION_TO_IDX["APPROVE"]] = -999.0
            q_vals[ACTION_TO_IDX["RETRY"]] = -999.0  # Fraud attacks are not retryable errors

        # Hard Guardrail 2: Blurry / glare image cannot be approved
        if features.get("doc_quality", 1.0) < 0.60:
            q_vals[ACTION_TO_IDX["APPROVE"]] = -999.0

        best_action_idx = int(np.argmax(q_vals))
        best_action = IDX_TO_ACTION[best_action_idx]

        return {
            "action": best_action,
            "q_values": {ACTIONS[i]: round(float(q_vals[i]), 3) for i in range(len(ACTIONS))},
            "state_key": state,
            "policy_type": "hybrid_rules_plus_qlearning"
        }

