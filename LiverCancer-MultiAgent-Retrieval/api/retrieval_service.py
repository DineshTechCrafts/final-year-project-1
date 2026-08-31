import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.evidence_validator import EvidenceValidator
from agents.explanation_agent import ExplanationAgent

class RetrievalService:
    def __init__(self):
        self.validator = EvidenceValidator()
        self.explainer = ExplanationAgent()
        
    def generate_report(self, query_case_id: str, candidate_evidence: dict) -> dict:
        if not self.validator.validate(candidate_evidence):
            return {
                "status": "EXPLANATION_UNAVAILABLE",
                "reason": "Evidence failed mathematical validation boundaries."
            }
            
        explanation = self.explainer.generate(candidate_evidence)
        
        return {
            "status": "SUCCESS",
            "candidate_case_id": candidate_evidence["candidate_case_id"],
            "explanation": explanation
        }
