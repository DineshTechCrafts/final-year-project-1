import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from preprocessing.hf_dataset_client import HFDatasetClient

def main():
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    cache_cases_dir = PROJECT_ROOT / "data" / "cache" / "cases"
    cache_cases_dir.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_dir / "patient_index.json", "r") as f:
        patient_index = json.load(f)
        
    client = HFDatasetClient("MedOtter/HCC-TACE-Seg", cache_dir=PROJECT_ROOT / "data" / "cache")
    
    # Pick first 3 patients for Stage 2 validation
    test_patients = list(patient_index.keys())[:3]
    
    stage2_data_structure_report = {
        "dimensionality": "Slice-based 2D (reconstructable to 3D pixel-space volume)",
        "image_format": "JPEG (8-bit, 0-255, non-HU windowed)",
        "mask_format": "JPEG (RGB, 256 artifact colors)",
        "spatial_information": "Not available (no voxel spacing, no thickness, no orientation)"
    }
    
    valid_cases = []
    invalid_cases = []
    mask_labels_observed = set()
    
    total_slices = 0
    img_shapes = set()
    mask_shapes = set()
    
    for pid in test_patients:
        row_indices = patient_index[pid]
        
        slices_data = []
        for r_idx in row_indices:
            batch = client.get_cached_rows("default", "preview", r_idx, 1)
            row_data = batch["rows"][0]["row"]
            
            # Fetch image
            img_url = row_data.get("image", {}).get("src")
            img_arr = None
            if img_url:
                resp = requests.get(img_url)
                img = Image.open(BytesIO(resp.content)).convert("L")
                img_arr = np.array(img)
                
            # Fetch mask
            mask_url = row_data.get("mask", {}).get("src")
            mask_arr = None
            if mask_url:
                resp = requests.get(mask_url)
                mask = Image.open(BytesIO(resp.content)).convert("RGB")
                mask_arr = np.array(mask)
            
            slices_data.append({
                "slice_index": row_data.get("slice_index", 0),
                "image": img_arr,
                "mask": mask_arr,
                "ct_series_uid": row_data.get("ct_series_uid")
            })
            
        # Group by ct_series_uid
        series_groups = {}
        for s in slices_data:
            suid = s["ct_series_uid"]
            if suid not in series_groups:
                series_groups[suid] = []
            series_groups[suid].append(s)
            
        for suid, group_slices in series_groups.items():
            case_id = f"{pid}_{suid[-6:]}"
            
            # Sort by slice index
            group_slices.sort(key=lambda x: x["slice_index"])
            
            # Verify consistency
            is_valid = True
            reasons = []
            
            volume_img = []
            volume_mask = []
            
            for s in group_slices:
                if s["image"] is None or s["mask"] is None:
                    is_valid = False
                    reasons.append("Missing image or mask data")
                    break
                
                if s["image"].shape[:2] != s["mask"].shape[:2]:
                    is_valid = False
                    reasons.append("Dimension mismatch")
                    break
                    
                volume_img.append(s["image"])
                volume_mask.append(s["mask"])
                img_shapes.add(s["image"].shape)
                mask_shapes.add(s["mask"].shape)
                
                # Analyze mask labels
                mask_labels_observed.update(np.unique(s["mask"]).tolist())
                
            if not is_valid:
                invalid_cases.append({"case_id": case_id, "reasons": reasons})
                continue
                
            # Stack into 3D volume
            vol_img_np = np.stack(volume_img, axis=0)
            vol_mask_np = np.stack(volume_mask, axis=0)
            
            # Save reconstructed case
            case_dir = cache_cases_dir / case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            np.save(case_dir / "image.npy", vol_img_np)
            np.save(case_dir / "mask.npy", vol_mask_np)
            
            case_metadata = {
                "internal_patient_id": pid,
                "internal_case_id": case_id,
                "ct_series_uid": suid,
                "num_slices": len(group_slices),
                "image_shape": vol_img_np.shape,
                "mask_shape": vol_mask_np.shape,
                "intensity_representation": "Windowed 8-bit JPEG (not HU)",
                "spatial_metadata": None
            }
            with open(case_dir / "metadata.json", "w") as f:
                json.dump(case_metadata, f, indent=4)
                
            valid_cases.append(case_id)
            total_slices += len(group_slices)

    # Output Reports
    with open(metadata_dir / "stage2_data_structure_report.json", "w") as f:
        json.dump(stage2_data_structure_report, f, indent=4)
        
    case_validation_report = {
        "valid_cases": valid_cases,
        "invalid_cases": invalid_cases
    }
    with open(metadata_dir / "case_validation_report.json", "w") as f:
        json.dump(case_validation_report, f, indent=4)
        
    mask_label_report = {
        "unique_values_observed": sorted(list(mask_labels_observed)),
        "issue": "Mask is stored as RGB JPEG. Values are continuous (0-255) due to compression artifacts. Needs color thresholding."
    }
    with open(metadata_dir / "mask_label_report.json", "w") as f:
        json.dump(mask_label_report, f, indent=4)
        
    stage2_summary = {
        "total_cases_processed": len(valid_cases) + len(invalid_cases),
        "total_valid_cases": len(valid_cases),
        "total_invalid_cases": len(invalid_cases),
        "total_slices_reconstructed": total_slices,
        "image_shape_distribution": [str(s) for s in list(img_shapes)],
        "mask_shape_distribution": [str(s) for s in list(mask_shapes)],
        "spatial_metadata_availability": "None"
    }
    with open(metadata_dir / "stage2_summary.json", "w") as f:
        json.dump(stage2_summary, f, indent=4)
        
    print("STAGE 2 PIPELINE COMPLETE.")

if __name__ == "__main__":
    main()
