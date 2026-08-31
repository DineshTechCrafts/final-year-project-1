import numpy as np
import pandas as pd

class EnhancementAgent:
    def __init__(self, scales: dict = None):
        self.scales = scales or {
            "intensity": 50.0
        }
        
    def evaluate(self, q_features: pd.Series, c_features: pd.Series) -> dict:
        if q_features.get("tumor_present", 1) == 0 or c_features.get("tumor_present", 1) == 0:
            return {
                "agent": "Enhancement",
                "similarity_score": np.nan,
                "confidence": 0.0,
                "missing_evidence": True,
                "reason": "Tumor absent."
            }
            
        mean_q = q_features.get("tumor_mean_intensity", 0)
        mean_c = c_features.get("tumor_mean_intensity", 0)
        
        if pd.isna(mean_q) or pd.isna(mean_c):
            return {
                "agent": "Enhancement",
                "similarity_score": np.nan,
                "confidence": 0.0,
                "missing_evidence": True,
                "reason": "Intensity features are NaN."
            }
            
        diff = abs(mean_q - mean_c)
        score = np.exp(-diff / self.scales["intensity"])
        
        return {
            "agent": "Enhancement",
            "similarity_score": float(score),
            "confidence": 1.0,
            "missing_evidence": False,
            "reason": f"8-bit image-intensity similarity: {score:.2f}"
        }
