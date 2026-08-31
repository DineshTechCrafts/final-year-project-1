import numpy as np
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from features.morphology import extract_morphology
from features.anatomy import extract_anatomy
from features.intensity import extract_intensity, extract_peritumoral_intensity

def test_morphology_extraction_synthetic():
    mask = np.zeros((100, 100), dtype=np.uint8)
    # Create a square tumor 10x10 at (20,20)
    mask[20:30, 20:30] = 2
    
    features = extract_morphology(mask, 2)
    
    assert features["area_px"] == 100
    assert features["bbox_height"] == 10
    assert features["bbox_width"] == 10
    assert features["bbox_aspect_ratio"] == 1.0
    # Centroid of [20:30, 20:30] is 24.5
    assert np.isclose(features["centroid_x"], 24.5)
    assert np.isclose(features["centroid_y"], 24.5)

def test_missing_tumor_morphology():
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    features = extract_morphology(mask, 2)
    
    assert features["area_px"] == 0
    assert features["component_count"] == 0
    assert np.isnan(features["centroid_x"])

def test_anatomy_extraction():
    # Liver
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[0:50, 0:50] = 1
    # Tumor
    mask[10:20, 10:20] = 2
    
    morph_tumor = extract_morphology(mask, 2)
    morph_liver = extract_morphology(mask, 1)
    
    anat = extract_anatomy(morph_liver, morph_tumor, {"area_px": 0}, {"area_px": 0}, (100, 100))
    
    # 100 area / 2500 area
    assert np.isclose(anat["tumor_to_liver_area_ratio"], 100 / 2500)
    # Centroid distance
    # Tumor centroid (14.5, 14.5), Liver centroid (24.5, 24.5) => dx = 10, dy = 10
    # Normalized by 100 => 0.1, 0.1
    dist = np.sqrt(0.1**2 + 0.1**2)
    assert np.isclose(anat["tumor_to_liver_centroid_distance"], dist)
    
    assert np.isnan(anat["tumor_to_portal_centroid_distance"])

def test_intensity_extraction():
    img = np.zeros((100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    img[10:20, 10:20] = 120
    mask[10:20, 10:20] = 2
    
    features = extract_intensity(img, mask, 2)
    
    assert np.isclose(features["mean"], 120.0)
    assert np.isclose(features["min"], 120.0)
    assert np.isclose(features["max"], 120.0)

def test_peritumoral_intensity():
    img = np.zeros((100, 100), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    
    img[10:20, 10:20] = 200 # Tumor
    mask[10:20, 10:20] = 2
    
    # Peritumoral ring
    img[5:25, 5:25] = 50 
    img[10:20, 10:20] = 200
    
    pt_feat = extract_peritumoral_intensity(img, mask, 2, radius=2)
    
    assert np.isclose(pt_feat["peritumoral_mean"], 50.0)
