import faiss
import numpy as np
import pandas as pd
from typing import List, Dict

class CaseRetriever:
    def __init__(self, index: faiss.IndexFlatIP, metadata: pd.DataFrame):
        self.index = index
        self.metadata = metadata
        
        # Verify Integrity Invariant
        assert self.index.ntotal == len(self.metadata), "CRITICAL: Index vector count does not match metadata row count."
        
    def search_by_embedding(self, query_vector: np.ndarray, top_k: int = 10, top_n: int = 100, 
                            exclude_query_case_id: str = None, 
                            exclude_patient_id: str = None) -> List[Dict]:
        """
        Executes a FAISS search and performs Post-Filtering.
        """
        # Ensure correct shape and type
        q = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        
        # FAISS exact search
        distances, indices = self.index.search(q, top_n)
        
        results = []
        rank = 1
        
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1: # Not enough vectors
                continue
                
            sim = distances[0][i]
            row = self.metadata.iloc[idx]
            
            # Post-Filter: Exclude self-case
            if exclude_query_case_id and row['case_id'] == exclude_query_case_id:
                continue
                
            # Post-Filter: Exclude same-patient
            if exclude_patient_id and row['patient_id'] == exclude_patient_id:
                continue
                
            results.append({
                "rank": rank,
                "case_id": row['case_id'],
                "similarity": float(sim),
                "patient_id": row['patient_id'],
                "partition": row['partition']
            })
            
            rank += 1
            if len(results) >= top_k:
                break
                
        return results
        
    def search_by_case_id(self, case_id: str, top_k: int = 10, exclude_same_patient: bool = True) -> List[Dict]:
        """
        Looks up the embedding for a case_id, then executes the search.
        """
        match = self.metadata[self.metadata['case_id'] == case_id]
        if match.empty:
            raise ValueError(f"Case ID {case_id} not found in metadata.")
            
        idx = match.index[0]
        patient_id = match.iloc[0]['patient_id']
        
        # We need the actual vector. Since IndexFlatIP stores vectors natively, we can reconstruct it.
        query_vector = self.index.reconstruct(int(idx))
        
        exclude_pat = patient_id if exclude_same_patient else None
        
        return self.search_by_embedding(
            query_vector, 
            top_k=top_k, 
            top_n=100, 
            exclude_query_case_id=case_id, 
            exclude_patient_id=exclude_pat
        )
