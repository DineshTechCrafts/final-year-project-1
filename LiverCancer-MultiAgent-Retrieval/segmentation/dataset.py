import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

class HCCTACESegDataset(Dataset):
    def __init__(self, root_dir: str, split_file: str, split_name: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        with open(split_file, "r") as f:
            splits = json.load(f)
            
        self.patient_ids = splits.get(split_name, [])
        self.samples = []
        
        # Load all slices from these patients
        for pid in self.patient_ids:
            for case_dir in self.root_dir.glob(f"{pid}_*"):
                meta_path = case_dir / "metadata.json"
                if not meta_path.exists():
                    continue
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    
                # load npy volumes
                # volume shape: (num_slices, H, W)
                img_vol = np.load(case_dir / "image.npy")
                mask_vol = np.load(case_dir / "mask.npy")
                
                num_slices = meta["num_slices"]
                for i in range(num_slices):
                    self.samples.append({
                        "case_dir": case_dir,
                        "slice_index": i,
                        "patient_id": pid,
                        "case_id": meta["internal_case_id"],
                        "ct_series_uid": meta["ct_series_uid"]
                    })
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        case_dir = sample["case_dir"]
        s_idx = sample["slice_index"]
        
        # We load the full volumes and pick the slice. 
        # In a highly optimized pipeline, saving slices individually as PNG is faster.
        # But caching the NPY array is fine for this baseline.
        img_vol = np.load(case_dir / "image.npy")
        mask_vol = np.load(case_dir / "mask.npy")
        
        img_slice = img_vol[s_idx]
        mask_slice = mask_vol[s_idx]
        
        # Adapt Grayscale to 3-channel for ResNet34
        if len(img_slice.shape) == 2:
            img_slice = np.stack([img_slice, img_slice, img_slice], axis=-1)
            
        if self.transform:
            # Albumentations uses interpolation=cv2.INTER_NEAREST automatically for mask 
            # if we pass it as mask (but actually A.Compose uses linear for image and nearest for mask internally 
            # if we set it up properly, wait, albumentations uses INTER_NEAREST for masks by default).
            # To be absolutely sure, we can rely on Albumentations default mask behavior.
            augmented = self.transform(image=img_slice, mask=mask_slice)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to tensor
            image = torch.from_numpy(img_slice.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask_slice).long()
            
        # Ensure mask is long
        mask = mask.long()
        
        metadata = {
            "patient_id": sample["patient_id"],
            "case_id": sample["case_id"],
            "slice_index": s_idx
        }
        
        return image, mask, metadata
