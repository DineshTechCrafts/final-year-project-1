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
    
    with open(metadata_dir / "patient_split.json", "r") as f:
        patient_split = json.load(f)
        
    with open(metadata_dir / "patient_index.json", "r") as f:
        patient_index = json.load(f)
        
    client = HFDatasetClient("MedOtter/HCC-TACE-Seg", cache_dir=PROJECT_ROOT / "data" / "cache")
    
    print("Fetching all dataset rows efficiently in batches to avoid rate limits...")
    
    all_rows = []
    # 104 rows total. We can fetch 0-99 and 100-104
    try:
        b1 = client.get_cached_rows("default", "preview", 0, 100)
        all_rows.extend([r["row"] for r in b1["rows"]])
        b2 = client.get_cached_rows("default", "preview", 100, 100)
        all_rows.extend([r["row"] for r in b2["rows"]])
        print(f"Successfully fetched {len(all_rows)} row metadata items.")
    except Exception as e:
        print(f"Failed to fetch row metadata: {e}")
        return

    results = {
        "train": {"patients": 0, "slices": 0},
        "validation": {"patients": 0, "slices": 0},
        "test": {"patients": 0, "slices": 0},
    }
    
    total_processed = 0
    total_patients = sum(len(pids) for pids in patient_split.values())
    
    for split_name, pids in patient_split.items():
        for pid in pids:
            row_indices = patient_index.get(pid, [])
            
            slices_data = []
            for r_idx in row_indices:
                if r_idx >= len(all_rows):
                    continue
                row_data = all_rows[r_idx]
                
                try:
                    img_url = row_data.get("image", {}).get("src")
                    img_arr = None
                    if img_url:
                        resp = requests.get(img_url, timeout=10)
                        img = Image.open(BytesIO(resp.content)).convert("L")
                        img_arr = np.array(img)
                        
                    mask_url = row_data.get("mask", {}).get("src")
                    mask_arr = None
                    if mask_url:
                        resp = requests.get(mask_url, timeout=10)
                        mask = Image.open(BytesIO(resp.content)).convert("RGB")
                        mask_arr = np.array(mask)
                    
                    if img_arr is not None and mask_arr is not None and img_arr.shape[:2] == mask_arr.shape[:2]:
                        slices_data.append({
                            "slice_index": row_data.get("slice_index", 0),
                            "image": img_arr,
                            "mask": mask_arr,
                            "ct_series_uid": row_data.get("ct_series_uid")
                        })
                except Exception as e:
                    print(f"Error fetching image/mask for row {r_idx}: {e}")
                    continue
                
            series_groups = {}
            for s in slices_data:
                suid = s["ct_series_uid"]
                if suid not in series_groups:
                    series_groups[suid] = []
                series_groups[suid].append(s)
                
            patient_slices = 0
            for suid, group_slices in series_groups.items():
                case_id = f"{pid}_{suid[-6:]}"
                group_slices.sort(key=lambda x: x["slice_index"])
                
                vol_img_np = np.stack([s["image"] for s in group_slices], axis=0)
                vol_mask_np = np.stack([s["mask"] for s in group_slices], axis=0)
                
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
                    
                patient_slices += len(group_slices)
                
            if patient_slices > 0:
                results[split_name]["patients"] += 1
                results[split_name]["slices"] += patient_slices
                
            total_processed += 1
            if total_processed % 10 == 0:
                print(f"Processed {total_processed}/{total_patients} patients...")

    print("\nRECOVERY SUMMARY:")
    for split_name, stats in results.items():
        print(f"{split_name.upper()}: {stats['patients']} patients, {stats['slices']} slices")

if __name__ == "__main__":
    main()
