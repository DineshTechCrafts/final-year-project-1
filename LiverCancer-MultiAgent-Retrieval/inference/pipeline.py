import time
import json
import uuid
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.modality_router import ModalityRouter
from inference.input_adapter import InputAdapter
from inference.segmentation_adapter import SegmentationAdapter
from api.retrieval_service import RetrievalService

def run_inference(input_path: str, modality: str) -> dict:
    start_time = time.time()
    
    # 1. Modality Routing
    router = ModalityRouter()
    route_status = router.route(modality)
    
    if route_status == "MRI_PIPELINE_NOT_VALIDATED":
        return {
            "status": "ERROR",
            "message": "MRI_PIPELINE_NOT_VALIDATED",
            "warnings": ["MRI segmentation and retrieval models have not undergone rigorous frozen testing. Inference aborted."]
        }
    if route_status != "CT_PIPELINE_OK":
        return {
            "status": "ERROR",
            "message": "UNSUPPORTED_MODALITY"
        }
        
    query_id = f"Q_{uuid.uuid4().hex[:8]}"
    
    # 2. Input Adapter Validation
    input_adapter = InputAdapter()
    input_info = input_adapter.validate(input_path)
    if input_info["status"] != "success":
        return {
            "status": "ERROR",
            "message": input_info["message"]
        }
        
    # 3. Architectural Fail-Fast: Try loading the massive weights
    try:
        seg_adapter = SegmentationAdapter()
        # This will explicitly throw FileNotFoundError in the prototype environment
        seg_result = seg_adapter.run(input_info)
    except FileNotFoundError as e:
        return {
            "status": "ERROR",
            "message": str(e)
        }
    
    # Normally we would continue processing...
    return {
        "status": "SUCCESS",
        "query_case_id": query_id,
        "modality": modality,
        "processing_latency_sec": round(time.time() - start_time, 3),
        "results": []
    }
