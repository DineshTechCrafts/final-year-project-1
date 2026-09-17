#!/usr/bin/env python3
"""
Convert preprocessed PNG slices into volumetric .npy arrays and metadata.json
for HCCTACESegDataset.

Input:
  - data/processed/<patient>/images/*.png (windowed CT images, 256x256 uint8)
  - data/processed/<patient>/masks/*.png (class index masks 0-4, 256x256 uint8)
  - data/verified_download_plan.json (mapping of HCC_XXX to CT series UID)

Output:
  - LiverCancer-MultiAgent-Retrieval/data/cache/cases/<case_id>/
      image.npy (num_slices, 256, 256) uint8
      mask.npy (num_slices, 256, 256) uint8
      metadata.json (num_slices, internal_case_id, ct_series_uid, etc.)
"""

import json
import hashlib
import shutil
from pathlib import Path
import numpy as np
from PIL import Image

def main():
    workspace_root = Path(r"C:\Users\DINESH\Desktop\Liver wala thing")
    plan_path = workspace_root / "data" / "verified_download_plan.json"
    processed_dir = workspace_root / "data" / "processed"
    target_cache_dir = workspace_root / "LiverCancer-MultiAgent-Retrieval" / "data" / "cache" / "cases"
    backup_dir = workspace_root / "LiverCancer-MultiAgent-Retrieval" / "data" / "cache" / "cases_preview_backup"
    
    print("=" * 60)
    print("RECONCILING DATASET FOR HCCTACESegDataset")
    print("=" * 60)
    print(f"Plan path:       {plan_path}")
    print(f"Processed dir:   {processed_dir}")
    print(f"Target cases:    {target_cache_dir}")
    
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed dir not found: {processed_dir}")
        
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)
        
    # Backup old preview cases if not already backed up
    if target_cache_dir.exists() and not backup_dir.exists():
        print(f"\nBacking up old preview cases to: {backup_dir.name} ...")
        shutil.copytree(target_cache_dir, backup_dir)
        print("Backup complete.")
        
    target_cache_dir.mkdir(parents=True, exist_ok=True)
    
    converted_cases = 0
    total_slices = 0
    global_label_counts = {}
    skipped = []
    
    print("\nStarting conversion...")
    for idx, item in enumerate(plan, 1):
        p_name = item["patient"]
        status = item.get("status")
        if status != "VERIFIED":
            skipped.append((p_name, f"Status: {status}"))
            continue
            
        p_dir = processed_dir / p_name.lower()
        if not p_dir.exists():
            matched_dirs = [d for d in processed_dir.iterdir() if d.is_dir() and d.name.lower() == p_name.lower()]
            if matched_dirs:
                p_dir = matched_dirs[0]
            else:
                skipped.append((p_name, "Processed directory missing"))
                continue
                
        img_files = sorted((p_dir / "images").glob("*.png"))
        mask_files = sorted((p_dir / "masks").glob("*.png"))
        
        if not img_files or len(img_files) != len(mask_files):
            skipped.append((p_name, f"Mismatched images ({len(img_files)}) and masks ({len(mask_files)})"))
            continue
            
        pid = hashlib.sha256(p_name.encode()).hexdigest()[:12]
        ct_info = item.get("ct") or {}
        suid = ct_info.get("series_uid") or "000000"
        suid_suffix = suid[-6:] if len(suid) >= 6 else suid
        case_id = f"{pid}_{suid_suffix}"
        
        imgs = [np.array(Image.open(f).convert("L"), dtype=np.uint8) for f in img_files]
        masks = [np.array(Image.open(f).convert("L"), dtype=np.uint8) for f in mask_files]
        
        img_vol = np.stack(imgs, axis=0)      # (N, H, W) uint8
        mask_vol = np.stack(masks, axis=0)    # (N, H, W) uint8
        
        case_dir = target_cache_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(case_dir / "image.npy", img_vol)
        np.save(case_dir / "mask.npy", mask_vol)
        
        u_labels, counts = np.unique(mask_vol, return_counts=True)
        for u in u_labels:
            global_label_counts[int(u)] = global_label_counts.get(int(u), 0) + 1
            
        meta = {
            "internal_patient_id": pid,
            "internal_case_id": case_id,
            "ct_series_uid": suid,
            "num_slices": len(imgs),
            "image_shape": list(img_vol.shape),
            "mask_shape": list(mask_vol.shape),
            "original_patient_id": p_name,
            "labels_present": [int(u) for u in u_labels]
        }
        with open(case_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
            
        converted_cases += 1
        total_slices += len(imgs)
        
        if converted_cases % 20 == 0 or converted_cases == 104:
            print(f"  [{converted_cases}/104] Converted {p_name} -> {case_id} ({len(imgs)} slices, labels={meta['labels_present']})")

    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Successfully converted: {converted_cases} cases")
    print(f"Total slices converted: {total_slices}")
    print(f"Skipped patients:       {len(skipped)}")
    if skipped:
        for p, r in skipped:
            print(f"  - {p}: {r}")
    print(f"Cases with each label present: {global_label_counts}")
    print("=" * 60)

if __name__ == "__main__":
    main()
