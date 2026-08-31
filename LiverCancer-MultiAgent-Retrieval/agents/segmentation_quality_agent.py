import numpy as np
import pandas as pd

class SegmentationQualityAgent:
    def evaluate(self, q_features: pd.Series, c_features: pd.Series) -> dict:
        # Quality can just be based on uncertain pixel percentage
        q_uncert = q_features.get("uncertain_pixel_percentage", 0.0)
        c_uncert = c_features.get("uncertain_pixel_percentage", 0.0)
        
        # Lower uncertainty is better
        # A simple bounded similarity metric
        q_score = 1.0 - min(1.0, q_uncert)
        c_score = 1.0 - min(1.0, c_uncert)
        
        final_score = float(np.mean([q_score, c_score]))
        confidence = final_score
        
        return {
            "agent": "SegmentationQuality",
            "similarity_score": final_score,
            "confidence": confidence,
            "missing_evidence": False,
            "reason": f"Combined segmentation confidence: {final_score:.2f}"
        }
