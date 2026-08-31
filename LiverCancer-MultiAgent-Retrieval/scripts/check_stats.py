import json
import torch
from pathlib import Path

PROJECT_ROOT = Path('.')

with open(PROJECT_ROOT / 'data/metadata/patient_split.json', 'r') as f:
    splits = json.load(f)

slice_counts = {'train': 0, 'validation': 0, 'test': 0}

for split_name, patient_ids in splits.items():
    for pid in patient_ids:
        # Search cache cases for this patient
        for meta_path in (PROJECT_ROOT / 'data/cache/cases').glob('*/metadata.json'):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta['internal_patient_id'] == pid:
                slice_counts[split_name] += meta['num_slices']

print(f"Train Slices: {slice_counts['train']}")
print(f"Val Slices: {slice_counts['validation']}")
print(f"Test Slices: {slice_counts['test']}")

print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
