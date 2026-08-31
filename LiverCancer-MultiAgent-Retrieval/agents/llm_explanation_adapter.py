class LLMExplanationAdapter:
    """
    Optional LLM Wrapper. Disabled by default to prevent hallucination.
    """
    def __init__(self, enabled=False):
        self.enabled = enabled
        
    def verbalize(self, evidence: dict):
        if not self.enabled:
            return None
            
        raise NotImplementedError("LLM integration disabled in Stage 10 prototype.")
