import pytest
import numpy as np
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from segmentation.mask_reconstructor import MaskReconstructor

def test_mask_reconstruction_determinism(tmp_path):
    # 1. Create a dummy config
    config_path = tmp_path / "mask_labels.yaml"
    config_data = {
        "colors": {
            "black": {"rgb": [0, 0, 0], "label": 0},
            "green": {"rgb": [0, 255, 0], "label": 1},
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
        
    # 2. Create a dummy RGB mask
    rgb_mask = np.zeros((100, 100, 3), dtype=np.uint8)
    rgb_mask[10:50, 10:50] = [0, 250, 0] # Near green
    rgb_mask[60:90, 60:90] = [128, 128, 128] # Highly uncertain
    
    # 3. Instantiate and run twice
    reconstructor = MaskReconstructor(str(config_path))
    
    recon1, uncert1 = reconstructor.reconstruct(rgb_mask, threshold=20.0)
    recon2, uncert2 = reconstructor.reconstruct(rgb_mask, threshold=20.0)
    
    # 4. Check absolute byte-equivalence
    assert np.array_equal(recon1, recon2), "Reconstructed masks are not deterministic!"
    assert np.array_equal(uncert1, uncert2), "Uncertainty masks are not deterministic!"
    
    # 5. Spot-check behavior
    assert recon1[20, 20] == 1 # Assigned to green
    assert uncert1[20, 20] == 0 # Confident
    
    assert recon1[70, 70] == 255 # Uncertain
    assert uncert1[70, 70] == 255 # Marked as uncertain mask
