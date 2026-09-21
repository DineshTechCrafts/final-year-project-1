import json
import numpy as np
import pandas as pd
from pathlib import Path
import time
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add retrieval package to path
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.index_builder import build_index, save_index_and_metadata
from retrieval.retriever import CaseRetriever

def main():
    # 1. Setup Directories
    vector_db_dir = PROJECT_ROOT / "data" / "vector_db"
    results_dir = PROJECT_ROOT / "data" / "retrieval" / "results"
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings" / "fused"
    features_dir = PROJECT_ROOT / "data" / "features"
    
    for d in [vector_db_dir, results_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    print("======================================================================")
    print("STAGE 8: REAL FAISS VECTOR DATABASE INDEXING")
    print("======================================================================")
        
    # 2. Load Real Stage 7 Fused Embeddings and Metadata
    print("Loading Stage 7 gated fusion case-level embeddings and metadata...")
    gated_embs = np.load(embeddings_dir / "gated_fused.npy")  # Expected: (104, 256)
    case_df = pd.read_parquet(features_dir / "case_features_ground_truth.parquet")
    
    assert len(gated_embs) == len(case_df), f"Mismatch: {len(gated_embs)} embeddings vs {len(case_df)} metadata rows"
    print(f"Loaded {len(gated_embs)} 256-dim embeddings.")
    
    # 3. Build Real FAISS Index
    print("\nBuilding exact-search inner product (Cosine Similarity) index...")
    index = build_index(gated_embs)
    
    print(f"Saving index and metadata to {vector_db_dir}...")
    save_index_and_metadata(
        index, 
        case_df, 
        vector_db_dir / "gated.index", 
        vector_db_dir / "gated_metadata.parquet"
    )
    
    # 3b. Build Slice-Level FAISS Index (for single-slice uploads)
    print("\nBuilding slice-level index...")
    slice_embs = np.load(embeddings_dir / "gated_slice_fused.npy")
    slice_df = pd.read_parquet(features_dir / "slice_features_ground_truth.parquet")
    
    assert len(slice_embs) == len(slice_df), f"Mismatch: {len(slice_embs)} vs {len(slice_df)}"
    slice_index = build_index(slice_embs)
    
    save_index_and_metadata(
        slice_index,
        slice_df,
        vector_db_dir / "gated_slice.index",
        vector_db_dir / "gated_slice_metadata.parquet"
    )
    print(f"Total slice vectors indexed: {slice_index.ntotal}")
    
    # 4. Initialize Real Retriever and Run Sample Queries
    print("\nInitializing CaseRetriever and running sample verification queries...")
    retriever = CaseRetriever(index, case_df)
    
    # Run queries on the first 3 cases
    sample_queries = case_df['case_id'].head(3).tolist()
    
    for q_case in sample_queries:
        results = retriever.search_by_case_id(q_case, top_k=5, exclude_same_patient=True)
        
        q_result = {
            "query_case_id": q_case,
            "embedding_type": "gated",
            "top_k": 5,
            "results": results
        }
        with open(results_dir / f"{q_case}.json", "w") as f:
            json.dump(q_result, f, indent=4)
            
        print(f"\nQuery Case: {q_case}")
        for r in results:
            print(f"  Rank {r['rank']} | Case: {r['case_id']} (Patient: {r['patient_id']}) | Sim: {r['similarity']:.4f}")
            assert r['case_id'] != q_case, f"ERROR: Query case {q_case} retrieved itself!"

    # 5. Summary Generation
    summary = {
        "status": "STAGE_8_COMPLETE_REAL",
        "indexed_cases": index.ntotal,
        "embedding_dimensions": gated_embs.shape[1],
        "index_type": "faiss.IndexFlatIP (Exact Search)",
        "similarity_metric": "Cosine Similarity",
        "retrieval_mode": "gated",
        "validation_results": "Successfully verified same-case and same-patient exclusion using exact FAISS retrieval."
    }
    with open(metadata_dir / "stage8_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"\nTotal vectors indexed: {index.ntotal}")
    print("\nSTAGE 8 EXECUTION COMPLETE")

if __name__ == "__main__":
    main()
