class ExplanationAgent:
    """
    Deterministically generates the human-readable explanation from numerical structures.
    """
    def generate(self, evidence: dict) -> dict:
        scores = evidence["scores"]
        
        agent_scores = {
            "morphology": scores["morphology"],
            "anatomy": scores["anatomy"],
            "intensity": scores["intensity"]
        }
        
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_agents = [sorted_agents[0][0]]
        weakest = [sorted_agents[-1][0]]
        
        # Build explanation deterministically
        explanation = f"{evidence['candidate_case_id']} was ranked highly because its multimodal representation showed strong overall similarity to the query. "
        
        # Positive Evidence
        top_names = " and ".join([t.capitalize() for t in top_agents])
        explanation += f"Strongest similarity evidence came from {top_names}. "
        
        # Weak Evidence
        weak_names = " and ".join([w.capitalize() for w in weakest])
        explanation += f"In contrast, {weak_names} similarity was comparatively weaker. "
        
        # FAISS Disagreement
        if scores["faiss"] > 0.90 and sorted_agents[-1][1] < 0.60:
            explanation += f"Although the candidate had high dense-vector similarity, {weakest[0]}-derived evidence was substantially weaker, indicating disagreement between global visual representation and explicit tumor geometry. "
            
        confidence = "HIGH"
        if scores["segmentation_quality"] < 0.6:
            explanation += "Interpretation of segmentation-derived evidence is limited by elevated segmentation uncertainty."
            confidence = "LOW"
            
        return {
            "summary": explanation.strip(),
            "strongest_evidence": top_agents,
            "weakest_evidence": weakest,
            "explanation_confidence": confidence
        }
