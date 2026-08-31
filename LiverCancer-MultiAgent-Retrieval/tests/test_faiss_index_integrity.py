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

def test_faiss_index_integrity():
    # 5 vectors, dim 128
    embeddings = np.random.randn(5, 128).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms
    
    metadata = pd.DataFrame({
        "case_id": ["C1", "C2", "C3", "C4", "C5"],
        "patient_id": ["P1", "P2", "P3", "P4", "P5"],
        "partition": ["train", "train", "val", "test", "test"]
    })
    
    index = build_index(embeddings)
    retriever = CaseRetriever(index, metadata)
    
    assert index.ntotal == len(metadata)
    assert index.d == 128
    
    # Query C1, we expect C1 as top hit if we DON'T exclude it
    # We will test search_by_embedding without filters
    res = retriever.search_by_embedding(embeddings[0], top_k=5, top_n=5)
    
    # The raw FAISS nearest neighbor to C1 should be C1 (similarity 1.0)
    assert res[0]["case_id"] == "C1"
    assert np.isclose(res[0]["similarity"], 1.0, atol=1e-5)
    
    # The reconstructed vector should match exactly
    recon = index.reconstruct(0)
    assert np.allclose(recon, embeddings[0], atol=1e-6)
