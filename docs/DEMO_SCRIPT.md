# Live Demo & Presentation Script (Deliverable D13)

**Project ID:** HSR-AGAI-029 (Agentic Multimodal eKYC Verification, Risk Fusion and Escalation Agent)  
**Deliverable:** D13 – Live Demonstration Guide & Presentation Script  

This walkthrough follows the bank's presentation criteria, demonstrating:
1. **Case 1: Straight-Through Clean Approval** (`APPROVE`)
2. **Case 2: Degraded Optical Quality Recapture** (`RETRY`)
3. **Case 3: Coordinated Presentation Attack Escalation** (`REJECT via HITL`)
4. **Case 4: Human-in-the-Loop Override and Active Learning Feedback Loop**

---

## 1. Quick Startup Command
In a terminal, run:
```powershell
# Windows
.\run.bat

# Linux / macOS
./run.sh
```
Open your browser to: **`http://127.0.0.1:8000`**

---

## 2. Demonstration Flow

### Scene 1: Platform Overview & System Health
- **Spoken Narration:**
  > *"Welcome to our capstone demonstration of the Agentic Multimodal eKYC Verification and Risk Fusion Agent. Modeled after industry standards from FRS Labs, our system fuses computer vision, speech acoustics, document OCR, and device intelligence under an autonomous ReAct agent loop."*
- **Action:**
  - Show the top KPI banner: **99.0% Benchmark Accuracy**, **21.3 ms Average Latency**, and **0% False Acceptance Rate**.
  - Highlight the live agent status pill in the top right.

---

### Scene 2: Case 1 – Genuine Clean Identity (Straight-Through Approve)
- **Action:** Click the button **`✓ Clean Genuine (Approve)`** in the scenario banner.
- **Visuals on Screen:**
  - Case `BUNDLE-001` opens in the inspector.
  - Green **`APPROVE`** decision badge appears.
  - Risk score displays `0.088` (**LOW TIER**).
  - Media shows crisp ID card, live selfie, and verified 4-digit spoken audio.
  - The ReAct trace shows 7 autonomous reasoning hops completing in **~25 ms**.
- **Spoken Narration:**
  > *"Here, the applicant submitted valid credentials. The agent extracted OCR fields, validated face similarity (0.88), verified 100% liveness with no moiré artifacts, and checked acoustic challenge digits without replay distortion. The calibrated logistic fusion engine yielded a risk of 0.088, granting straight-through approval without requiring human intervention."*

---

### Scene 3: Case 2 – Optical Degradation & Early Stopping (User Retry)
- **Action:** Click the button **`📷 Blurry Document (Retry)`**.
- **Visuals on Screen:**
  - Case `BUNDLE-006` opens in the inspector.
  - Decision badge changes to amber **`RETRY`**.
  - Notice the **Total Steps = 2** and **Latency = 12 ms**.
- **Spoken Narration:**
  > *"In Case 2, the applicant uploaded a document suffering from severe motion blur and specular flash glare. Notice our ReAct stopping policy in action: rather than wasting compute running deep face embedding or acoustic models, the agent detects Laplacian variance under 500, queries regulatory policy POL-001, and issues an immediate RETRY action with user-friendly instructions to disable camera flash."*

---

### Scene 4: Case 3 – Presentation Spoof & Syndicate Fraud (Reject via HITL)
- **Action:** Click the button **`⚠️ Screen & Voice Spoof (Reject)`**.
- **Visuals on Screen:**
  - Case `BUNDLE-012` opens in the inspector.
  - Decision badge changes to crimson **`REJECT`**.
  - Risk score spikes to **`0.990`** (**CRITICAL TIER**).
  - Risk bars light up in red: Presentation Spoof (99%), Voice Replay (92%), and Device Collision.
  - Reviewer Queue badge updates to **`Escalated`**.
- **Spoken Narration:**
  > *"In Case 3, we simulate a coordinated presentation attack: a digital tablet displaying a captured photo, accompanied by a loudspeaker replay attack and a hardware ID belonging to a known fraud syndicate. The multi-spectral 2D FFT detects screen grid harmonics, and the acoustic analyzer identifies non-linear speaker clipping. Safety Guardrail POL-002 immediately quarantines the case to the Human Supervisor queue."*

---

### Scene 5: Case 4 – Human-In-The-Loop Review & Continuous Feedback
- **Action:**
  - In the left sidebar, click **`Reviewer Queue`** filter tab to show only escalated cases.
  - Scroll down to the **Human-In-The-Loop Supervisor Review Console**.
  - Enter notes: *"Confirmed screen raster lines and transducer resonance. Finalized rejection."*
  - Click **`✕ Reject as Fraudulent`**.
  - Observe the notification confirming the feedback has been logged to the active learning pipeline.
- **Spoken Narration:**
  > *"Our HITL console empowers compliance officers to inspect full replayable ReAct thought traces, listen to audio challenges, and log ground-truth overrides that continuously retrain our perception and fusion thresholds."*

