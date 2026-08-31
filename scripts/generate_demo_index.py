import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "demo_index.json"

def process_slice(args):
    patient_dir, img_file = args
    patient_id = patient_dir.name
    slice_idx = int(img_file.stem)
    
    img_path = patient_dir / "images" / img_file.name
    mask_path = patient_dir / "masks" / img_file.name
    
    if not img_path.exists() or not mask_path.exists():
        return None
        
    img_arr = np.array(Image.open(img_path).convert('L'))
    mask_arr = np.array(Image.open(mask_path).convert('L'))
    
    total_pixels = mask_arr.size
    
    liver_pixels = np.sum(mask_arr == 1)
    mass_pixels = np.sum(mask_arr == 2)
    portal_pixels = np.sum(mask_arr == 3)
    aorta_pixels = np.sum(mask_arr == 4)
    
    mean_intensity = float(np.mean(img_arr))
    std_intensity = float(np.std(img_arr))
    
    return {
        "patient_id": patient_id,
        "slice_index": slice_idx,
        "features": {
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "liver_ratio": float(liver_pixels / total_pixels),
            "mass_ratio": float(mass_pixels / total_pixels),
            "portal_ratio": float(portal_pixels / total_pixels),
            "aorta_ratio": float(aorta_pixels / total_pixels),
        }
    }

def main():
    print("Collecting slices...")
    tasks = []
    if PROCESSED_DATA_PATH.exists():
        for p_dir in PROCESSED_DATA_PATH.iterdir():
            if not p_dir.is_dir() or p_dir.name == "hcc055":
                continue
            images_dir = p_dir / "images"
            if images_dir.exists():
                for img_file in images_dir.glob("*.png"):
                    tasks.append((p_dir, img_file))
                    
    print(f"Found {len(tasks)} slices. Processing...")
    
    results = []
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(process_slice, tasks), total=len(tasks)):
            if res is not None:
                results.append(res)
                
    print(f"Saving {len(results)} records to index...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f)
    print("Done!")

if __name__ == "__main__":
    main()
