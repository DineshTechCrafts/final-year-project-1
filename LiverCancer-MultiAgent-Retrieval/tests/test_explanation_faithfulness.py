import pytest
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.explanation_agent import ExplanationAgent

def test_explanation_faithfulness():
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
    
    assert "CASE_C" in out["summary"]
    assert out["strongest_evidence"] == ["morphology"]
    assert out["weakest_evidence"] == ["intensity"]
    
    # Mathematical extraction works correctly
    assert "Strongest similarity evidence came from Morphology" in out["summary"]
    assert "Intensity similarity was comparatively weaker" in out["summary"]
