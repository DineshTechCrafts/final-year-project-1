import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import run_inference

def test_inference_blocks_mri():
    result = run_inference("mock_mri.jpg", "MRI")
    assert result["status"] == "ERROR"
    assert result["message"] == "MRI_PIPELINE_NOT_VALIDATED"

def test_inference_ct_success():
    result = run_inference("mock_ct.jpg", "CT")
    
    assert result["status"] == "SUCCESS"
    assert "query_case_id" in result
    assert len(result["results"]) > 0
    
    top = result["results"][0]
    # Check explanations have no fabricated strings
    banned = ["diagnosis", "malignant", "prognosis"]
    summary = top["explanation"]["summary"].lower()
    for b in banned:
        assert b not in summary
        
    print("End-to-end Inference Tests Passed!")

if __name__ == "__main__":
    test_inference_blocks_mri()
    test_inference_ct_success()
