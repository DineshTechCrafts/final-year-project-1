import numpy as np
import pandas as pd

class AnatomyAgent:
    def __init__(self, scales: dict = None):
        self.scales = scales or {
            "centroid_dist": 0.5
        }
        
    def evaluate(self, q_features: pd.Series, c_features: pd.Series) -> dict:
        if q_features.get("tumor_present", 1) == 0 or c_features.get("tumor_present", 1) == 0:
            return {
                "agent": "Anatomy",
                "similarity_score": np.nan,
                "confidence": 0.0,
                "missing_evidence": True,
                "reason": "Tumor absent."
            }
            
        # Example: compare tumor_to_liver_centroid_distance
        dist_q = q_features.get("tumor_to_liver_centroid_distance", 0)
        dist_c = c_features.get("tumor_to_liver_centroid_distance", 0)
        
        if pd.isna(dist_q) or pd.isna(dist_c):
            return {
                "agent": "Anatomy",
                "similarity_score": np.nan,
                "confidence": 0.0,
                "missing_evidence": True,
                "reason": "Centroid distances are NaN."
            }
            
        diff = abs(dist_q - dist_c)
        score = np.exp(-diff / self.scales["centroid_dist"])
        
        return {
            "agent": "Anatomy",
            "similarity_score": float(score),
            "confidence": 1.0,
            "missing_evidence": False,
            "reason": f"Anatomical position similarity: {score:.2f}"
        }
