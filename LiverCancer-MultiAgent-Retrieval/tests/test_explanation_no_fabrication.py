import pytest
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.explanation_agent import ExplanationAgent

def test_no_fabrication():
    agent = ExplanationAgent()
    evidence = {
        "query_case_id": "CASE_Q",
        "candidate_case_id": "CASE_C",
        "scores": {
            "faiss": 0.91,
            "morphology": 0.88,
            "anatomy": 0.82,
            "intensity": 0.73,
            "segmentation_quality": 0.94,
            "final": 0.87
        }
    }
    
    out = agent.generate(evidence)
    
    banned_words = [
        "diagnosis", "prognosis", "malignant", "benign", 
        "treatment", "survival", "recommendation"
    ]
    
    summary_lower = out["summary"].lower()
    
    for word in banned_words:
        assert word not in summary_lower, f"Fabricated term '{word}' found in explanation."
