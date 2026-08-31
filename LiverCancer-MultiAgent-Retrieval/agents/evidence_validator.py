import math

class EvidenceValidator:
    """
    Validates Stage 9 structured JSON outputs to ensure no hallucinated metrics are passed to the explainer.
    """
    def validate(self, candidate_evidence: dict) -> bool:
        scores = candidate_evidence.get("scores", {})
        
        required_keys = ["faiss", "morphology", "anatomy", "intensity", "segmentation_quality", "final"]
        for k in required_keys:
            if k not in scores:
                return False
                
            val = scores[k]
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                return False
                
            if not (0.0 <= val <= 1.0):
                return False
                
        # Candidate and Query IDs must exist
        if not candidate_evidence.get("candidate_case_id") or not candidate_evidence.get("query_case_id"):
            return False
            
        return True
