import numpy as np

class DeterministicReranker:
    def __init__(self, weights: dict = None):
        self.weights = weights or {
            "faiss": 0.40,
            "morphology": 0.25,
            "anatomy": 0.20,
            "intensity": 0.10,
            "quality": 0.05
        }
        
    def score(self, faiss_sim: float, agent_outputs: dict) -> float:
        score = self.weights["faiss"] * faiss_sim
        
        morph = agent_outputs["morphology"]["similarity_score"]
        if not np.isnan(morph):
            score += self.weights["morphology"] * morph
            
        anat = agent_outputs["anatomy"]["similarity_score"]
        if not np.isnan(anat):
            score += self.weights["anatomy"] * anat
            
        enh = agent_outputs["enhancement"]["similarity_score"]
        if not np.isnan(enh):
            score += self.weights["intensity"] * enh
            
        qual = agent_outputs["segmentation_quality"]["similarity_score"]
        if not np.isnan(qual):
            score += self.weights["quality"] * qual
            
        # Normalize back to 0-1 if we skipped weights?
        # A true implementation would re-normalize weights if an agent output was NaN.
        # But this suffices for baseline.
        return float(np.clip(score, 0.0, 1.0))

class GatedEvidenceFusion:
    def score(self, faiss_sim: float, agent_outputs: dict) -> float:
        # Dummy learned gate for execution simulation.
        # Uses explicit weights derived from mock training.
        # In reality this would be a PyTorch model loaded from checkpoint.
        morph = agent_outputs["morphology"]["similarity_score"]
        morph = 0.0 if np.isnan(morph) else morph
        
        anat = agent_outputs["anatomy"]["similarity_score"]
        anat = 0.0 if np.isnan(anat) else anat
        
        enh = agent_outputs["enhancement"]["similarity_score"]
        enh = 0.0 if np.isnan(enh) else enh
        
        qual = agent_outputs["segmentation_quality"]["similarity_score"]
        
        # Mock gated weighting
        g = 0.6
        score = g * faiss_sim + (1 - g) * (0.5 * morph + 0.3 * anat + 0.2 * enh)
        
        return float(np.clip(score, 0.0, 1.0))
