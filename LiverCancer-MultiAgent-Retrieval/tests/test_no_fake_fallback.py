import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.pipeline import run_inference

def test_no_fake_fallback_enforcement():
    # Attempting to run inference on a dummy path should immediately crash
    # and hit the strict FileNotFoundError block rather than returning a mocked CASE_001.
    
    # Create a dummy image
    dummy_img = PROJECT_ROOT / "tmp_test_img.jpg"
    with open(dummy_img, "w") as f:
        f.write("mock bytes")
        
    try:
        result = run_inference(str(dummy_img), "CT")
        
        # We explicitly demand the strict missing artifact failure
        assert result["status"] == "ERROR"
        assert result["message"] == "SEGMENTATION_ARTIFACT_UNAVAILABLE"
        
    finally:
        dummy_img.unlink(missing_ok=True)
        
    print("Passed Test: Pipeline strictly fails on missing PyTorch artifacts without falling back to synthetic stubs.")

if __name__ == "__main__":
    test_no_fake_fallback_enforcement()
