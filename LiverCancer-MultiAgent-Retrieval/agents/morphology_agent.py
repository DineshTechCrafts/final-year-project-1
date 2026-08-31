import numpy as np
import pandas as pd

class MorphologyAgent:
    def __init__(self, scales: dict = None):
        # Default scales determined exclusively from TRAIN
        self.scales = scales or {
            "area_px": 500.0,
            "bbox_aspect_ratio": 1.0,
            "solidity": 0.2
        }
        
    def evaluate(self, q_features: pd.Series, c_features: pd.Series) -> dict:
        """
        Evaluates explicit morphological similarities.
        """
        # If tumor absent in either, evidence is missing.
        if q_features.get("tumor_present", 1) == 0 or c_features.get("tumor_present", 1) == 0:
            return {
                "agent": "Morphology",
                "similarity_score": np.nan,
                "confidence": 0.0,
                "missing_evidence": True,
                "reason": "Tumor absent in one or both cases."
            }
            
        area_diff = abs(q_features.get("tumor_area_px", 0) - c_features.get("tumor_area_px", 0))
        area_sim = np.exp(-area_diff / self.scales["area_px"])
        
        # We can average the feature similarities
        final_score = float(area_sim)
        
        return {
            "agent": "Morphology",
            "similarity_score": final_score,
            "confidence": 1.0,
            "missing_evidence": False,
            "reason": f"Computed area similarity: {final_score:.2f}."
        }
