import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def test_patient_leakage():
    split_file = PROJECT_ROOT / "data" / "metadata" / "patient_split.json"
    if not split_file.exists():
        pytest.skip("patient_split.json not generated yet")
        
    with open(split_file, "r") as f:
        splits = json.load(f)
        
    train = set(splits.get("train", []))
    val = set(splits.get("validation", []))
    test = set(splits.get("test", []))
    
    # 1. Zero patient intersection
    assert len(train.intersection(val)) == 0, "Patient leakage between train and val"
    assert len(train.intersection(test)) == 0, "Patient leakage between train and test"
    assert len(val.intersection(test)) == 0, "Patient leakage between val and test"

    # In a full pipeline we would also assert that slices from the same CT series
    # never span across train/val/test directories in data/processed/segmentation_2d.
    # We will enforce this by building the dataset purely via patient-level iteration.
