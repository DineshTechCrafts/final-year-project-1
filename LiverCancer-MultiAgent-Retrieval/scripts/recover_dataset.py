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

# ---------------------------------------------------------------------------
# Color → class-index lookup table
#
# The HCC-TACE-Seg HuggingFace dataset renders segmentation masks as RGB PNGs
# using this fixed palette (Background is pure black = no annotation):
#
#   0 = Background  (  0,   0,   0)  — black
#   1 = Liver       (255,   0,   0)  — red
#   2 = HCC Mass    (255, 255,   0)  — yellow
#   3 = Portal Vein (  0,   0, 255)  — blue
#   4 = Aorta       (  0, 255,   0)  — green
#
# rgb_to_class() converts an (H, W, 3) uint8 RGB array →
# an (H, W) uint8 class-index array with values in {0, 1, 2, 3, 4}.
# ---------------------------------------------------------------------------

_COLOR_TO_CLASS = np.array([
    [  0,   0,   0],   # 0 = Background
    [255,   0,   0],   # 1 = Liver
    [255, 255,   0],   # 2 = HCC Mass
    [  0,   0, 255],   # 3 = Portal Vein
    [  0, 255,   0],   # 4 = Aorta
], dtype=np.float32)

# Maximum L2 distance (in RGB space) to still assign a class.
# sqrt(3 * 255^2) ≈ 441 is the maximum possible distance.
# We use 80 as a generous threshold that covers JPEG boundary smear
# while still being far enough from neighbouring palette colours to
# avoid mis-classification.
_MAX_PALETTE_DIST = 80.0


def rgb_to_class(rgb: np.ndarray) -> np.ndarray:
    """Map (H, W, 3) uint8 RGB mask -> (H, W) uint8 class-index array.

    Uses nearest-palette-colour (L2 distance) assignment so that JPEG
    compression artefacts at segment boundaries are still mapped to the
    correct class rather than silently falling through to Background.

    Pixels whose nearest palette colour is more than _MAX_PALETTE_DIST
    away in L2 RGB space are mapped to 0 (Background).
    """
    H, W, _ = rgb.shape
    # (H*W, 3) float array
    flat = rgb.reshape(-1, 3).astype(np.float32)

    # Broadcast subtraction: (H*W, 1, 3) - (1, 5, 3) -> (H*W, 5, 3)
    diff = flat[:, np.newaxis, :] - _COLOR_TO_CLASS[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))  # (H*W, 5)

    nearest_class = np.argmin(dists, axis=-1).astype(np.uint8)  # (H*W,)
    nearest_dist  = dists[np.arange(len(flat)), nearest_class]   # (H*W,)

    # Pixels too far from any known colour default to Background
    nearest_class[nearest_dist > _MAX_PALETTE_DIST] = 0

    return nearest_class.reshape(H, W)



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
                        # Download the RGB-rendered mask PNG and convert it
                        # back to a class-index label map (values 0-4).
                        mask_rgb = np.array(
                            Image.open(BytesIO(resp.content)).convert("RGB")
                        )
                        mask_arr = rgb_to_class(mask_rgb)  # shape (H, W), dtype uint8
                    
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
