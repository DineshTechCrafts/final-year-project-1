import time
import sys
from pathlib import Path
from fastapi import HTTPException

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_ROOT = PROJECT_ROOT / "LiverCancer-MultiAgent-Retrieval"

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from agents.decision_agent import DecisionResultsAgent, DecisionAgentError

# Keep a single instance to avoid re-parsing configs
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = DecisionResultsAgent()
    return _agent

def synthesize_pipeline_results(payload: dict) -> dict:
    try:
        agent = get_agent()
        
        # Build the exact evidence structure expected by the decision agent
        # The payload is the JSON returned by retrieval_service.process_uploaded_image
        
        results = payload.get("results", [])
        simulated_features = payload.get("simulated_features", {})
        
        tumor_detected = simulated_features.get("mass_ratio", 0.0) > 0
        
        evidence = {
            "patient_id": payload.get("query_case_id", "upload"),
            "imaging": {
                "modality": "CT",
                "slice_index": "N/A (Upload)"
            },
            "segmentation": {
                "tumor_detected": tumor_detected,
                "tumor_area_ratio": simulated_features.get("mass_ratio", 0.0),
                "liver_area_ratio": simulated_features.get("liver_ratio", 0.0),
                "segmentation_warning": payload.get("segmentation_warning")
            },
            "retrieval": {
                "number_of_cases": len(results),
                "similar_cases": []
            },
            "deep_features": {
                "embedding_model": "resnet50_3d + multimodal fusion",
                "embedding_available": True,
                "feature_summary": "Visual and structured embedding generated from query slice."
            },
            "radiomics": {
                "available": True,
                "shape_features": {},
                "first_order_features": {"mean_intensity": simulated_features.get("mean_intensity", 0.0)},
                "texture_features": {}
            },
            "clinical_context": {
                "available": False,
                "data": {}
            },
            "fusion": {
                "final_scores": {},
                "ranking_method": "multi-agent reranking (Stage 9)"
            }
        }
        
        # We must include the warning at the top level for the rule to catch it easily, or just in segmentation
        if payload.get("segmentation_warning"):
            evidence["segmentation_warning"] = payload["segmentation_warning"]
            
        for idx, c in enumerate(results):
            evidence["retrieval"]["similar_cases"].append({
                "case_id": c.get("case_id"),
                "slice_index": c.get("slice_index"),
                "similarity_score": c.get("similarity"),
                "faiss_rank": idx + 1,
            })
            
        t0 = time.perf_counter()
        llm_result = agent.synthesize(evidence)
        t1 = time.perf_counter()
        
        return {
            "synthesis": llm_result,
            "latency_seconds": round(t1 - t0, 2)
        }
    except DecisionAgentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Synthesis failed: {str(e)}")
