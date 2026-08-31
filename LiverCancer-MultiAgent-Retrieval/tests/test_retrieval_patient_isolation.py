import faiss
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.index_builder import build_index
from retrieval.retriever import CaseRetriever

def test_patient_isolation():
    # We will make identical embeddings for C1 and C2 (same patient)
    # C3 is different patient.
    
    embeddings = np.zeros((3, 128), dtype=np.float32)
    embeddings[0, 0] = 1.0
    embeddings[1, 0] = 1.0
    embeddings[2, 1] = 1.0
    
    metadata = pd.DataFrame({
        "case_id": ["C1", "C2", "C3"],
        "patient_id": ["P1", "P1", "P2"],
        "partition": ["train", "train", "test"]
    })
    
    index = build_index(embeddings)
    retriever = CaseRetriever(index, metadata)
    
    # 1. Search C1 WITH same patient exclusion
    res = retriever.search_by_case_id("C1", top_k=2, exclude_same_patient=True)
    
    # It should explicitly exclude C1 (itself) and C2 (same patient)
    assert len(res) == 1
    assert res[0]["case_id"] == "C3"
    
    # 2. Search C1 WITHOUT same patient exclusion
    res2 = retriever.search_by_case_id("C1", top_k=2, exclude_same_patient=False)
    
    # It should exclude C1 (itself) but RETURN C2
    assert len(res2) == 2
    assert res2[0]["case_id"] == "C2"
    assert res2[1]["case_id"] == "C3"
