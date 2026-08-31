import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.morphology_agent import MorphologyAgent
from agents.reranking_engine import DeterministicReranker

def test_missing_evidence():
    agent = MorphologyAgent()
    
    q_feat = pd.Series({"tumor_present": 0})
    c_feat = pd.Series({"tumor_present": 1, "tumor_area_px": 500})
    
    out = agent.evaluate(q_feat, c_feat)
    
    assert out["missing_evidence"] == True
    assert np.isnan(out["similarity_score"])
    
def test_score_bounds():
    agent = MorphologyAgent()
    
    q_feat = pd.Series({"tumor_present": 1, "tumor_area_px": 100})
    c_feat = pd.Series({"tumor_present": 1, "tumor_area_px": 100})
    
    out = agent.evaluate(q_feat, c_feat)
    assert np.isclose(out["similarity_score"], 1.0)
    
    c_feat2 = pd.Series({"tumor_present": 1, "tumor_area_px": 100000})
    out2 = agent.evaluate(q_feat, c_feat2)
    assert 0.0 <= out2["similarity_score"] <= 1.0

def test_reranker_bounds():
    reranker = DeterministicReranker()
    
    agents = {
        "morphology": {"similarity_score": 1.0},
        "anatomy": {"similarity_score": 1.0},
        "enhancement": {"similarity_score": 1.0},
        "segmentation_quality": {"similarity_score": 1.0}
    }
    
    score = reranker.score(1.0, agents)
    assert np.isclose(score, 1.0)
