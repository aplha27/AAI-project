"""
Agent Case History & Working Memory Store
Stores past cases, reviewer verdicts, fraud incident flags, and device associations.
"""

from typing import Dict, Any, List, Optional
import datetime

class CaseHistoryMemory:
    def __init__(self):
        self.cases: Dict[str, Dict[str, Any]] = {}
        self.device_case_map: Dict[str, List[str]] = {}
        self.identity_history: Dict[str, List[str]] = {}

    def record_case(self, case_id: str, case_data: Dict[str, Any]):
        self.cases[case_id] = case_data
        dev = case_data.get("applicant_data", {}).get("device_id")
        if dev:
            if dev not in self.device_case_map:
                self.device_case_map[dev] = []
            self.device_case_map[dev].append(case_id)

        id_num = case_data.get("applicant_data", {}).get("id_number")
        if id_num:
            if id_num not in self.identity_history:
                self.identity_history[id_num] = []
            self.identity_history[id_num].append(case_id)

    def query_history(self, device_id: Optional[str] = None, id_number: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves case history context for memory lookups.
        """
        related_cases = []
        repeat_device_count = 0
        previous_decisions = []

        if device_id and device_id in self.device_case_map:
            case_ids = self.device_case_map[device_id]
            repeat_device_count = len(case_ids)
            for cid in case_ids[-5:]:
                if cid in self.cases:
                    c = self.cases[cid]
                    related_cases.append({
                        "case_id": cid,
                        "applicant_name": c.get("applicant_data", {}).get("full_name"),
                        "decision": c.get("final_decision"),
                        "risk_tier": c.get("risk_tier")
                    })
                    previous_decisions.append(c.get("final_decision"))

        return {
            "device_id": device_id,
            "repeat_submissions_on_device": repeat_device_count,
            "previous_decisions": previous_decisions,
            "has_prior_rejections": "REJECT" in previous_decisions,
            "related_cases": related_cases
        }

