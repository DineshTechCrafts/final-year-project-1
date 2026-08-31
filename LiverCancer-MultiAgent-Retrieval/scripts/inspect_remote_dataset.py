"""
Script to inspect the remote HCC-TACE-Seg dataset via HF API.
Generates reports and deterministic patient-level splits.
"""
import os
import sys
import json
import hashlib
import random
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.hf_dataset_client import HFDatasetClient
from utils.logger import setup_logger

logger = setup_logger("inspect_remote_dataset")

def sanitize_id(raw_id: str) -> str:
    """Hash patient identifiers to avoid exposing PII unnecessarily."""
    return hashlib.sha256(str(raw_id).encode()).hexdigest()[:12]

def main():
    dataset_name = "MedOtter/HCC-TACE-Seg"
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    client = HFDatasetClient(dataset_name=dataset_name, cache_dir=PROJECT_ROOT / "data" / "cache")
    
    logger.info(f"Inspecting remote dataset: {dataset_name}")
    
    try:
        splits_info = client.get_splits()
        logger.info(f"Found {len(splits_info)} splits.")
    except Exception as e:
        logger.error(f"Failed to connect to dataset API: {e}")
        return

    # Assume we use the first available split for inspection
    if not splits_info:
        logger.error("No splits found.")
        return
        
    target_config = splits_info[0]["config"]
    target_split = splits_info[0]["split"]
    
    # Retrieve size info
    size_info = client.get_size(target_config)
    
    total_rows = 0
    # Locate the total rows for the specific split
    for split_size in size_info.get("size", {}).get("splits", []):
        if split_size["split"] == target_split:
            total_rows = split_size["num_rows"]
            break
            
    logger.info(f"Target split '{target_split}' has {total_rows} rows.")
    
    # Paginate through all rows
    page_size = 100
    all_rows_info = []
    
    for offset in range(0, total_rows, page_size):
        length = min(page_size, total_rows - offset)
        logger.info(f"Fetching rows {offset} to {offset + length - 1}...")
        
        batch = client.get_cached_rows(target_config, target_split, offset, length)
        
        # Ensure we have the fields we expect
        if "rows" not in batch:
            logger.warning(f"Batch offset {offset} did not return 'rows'.")
            continue
            
        for item in batch["rows"]:
            row_idx = item["row_idx"]
            row_data = item["row"]
            
            raw_patient_id = row_data.get("patient_id", f"unknown_{row_idx}")
            internal_patient_id = sanitize_id(raw_patient_id)
            
            all_rows_info.append({
                "row_idx": row_idx,
                "internal_patient_id": internal_patient_id,
                "study_uid": row_data.get("study_uid"),
                "ct_series_uid": row_data.get("ct_series_uid"),
                "num_ct_slices": row_data.get("num_ct_slices"),
                "liver_voxels": row_data.get("liver_voxels"),
                "mass_voxels": row_data.get("mass_voxels"),
                "has_image": "image" in row_data and row_data["image"] is not None,
                "has_mask": "mask" in row_data and row_data["mask"] is not None
            })

    # Generate patient index
    patient_index = {}
    for info in all_rows_info:
        pid = info["internal_patient_id"]
        if pid not in patient_index:
            patient_index[pid] = []
        patient_index[pid].append(info["row_idx"])
        
    # Write patient index
    with open(metadata_dir / "patient_index.json", "w") as f:
        json.dump(patient_index, f, indent=4)
        
    unique_patients = list(patient_index.keys())
    unique_patients.sort() # Ensure deterministic order before shuffle
    
    logger.info(f"Identified {len(unique_patients)} unique patients.")
    
    # Perform deterministic split (70/15/15)
    random.seed(42)
    random.shuffle(unique_patients)
    
    n_patients = len(unique_patients)
    n_train = int(n_patients * 0.70)
    n_val = int(n_patients * 0.15)
    
    train_patients = unique_patients[:n_train]
    val_patients = unique_patients[n_train:n_train + n_val]
    test_patients = unique_patients[n_train + n_val:]
    
    patient_split = {
        "train": train_patients,
        "validation": val_patients,
        "test": test_patients
    }
    
    # Write patient split
    with open(metadata_dir / "patient_split.json", "w") as f:
        json.dump(patient_split, f, indent=4)
        
    # Generate remote dataset report
    first_batch = client.get_cached_rows(target_config, target_split, 0, 1)
    features = first_batch.get("features", [])
    
    report = {
        "dataset_name": dataset_name,
        "configuration": target_config,
        "split": target_split,
        "number_of_rows": total_rows,
        "unique_patients": len(unique_patients),
        "split_sizes": {
            "train": len(train_patients),
            "validation": len(val_patients),
            "test": len(test_patients)
        },
        "available_columns": [f["name"] for f in features],
        "image_fields": [f["name"] for f in features if f.get("type", {}).get("_type") == "Image"],
        "mask_fields": [f["name"] for f in features if f.get("name") in ["mask", "overlay"]]
    }
    
    with open(metadata_dir / "remote_dataset_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("Inspection complete. Generated report, index, and split manifests.")
    print("\n=========================================================================================================")
    print("STAGE 1 COMPLETE — HCC-TACE-Seg is accessible through the Hugging Face Dataset Server API without downloading the complete dataset.")
    print("=========================================================================================================\n")

if __name__ == "__main__":
    main()
