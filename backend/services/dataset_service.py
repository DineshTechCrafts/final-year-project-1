import json
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_SPLIT_PATH = PROJECT_ROOT / "data" / "dataset" / "split_info.json"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

class DatasetService:
    def __init__(self):
        self.split_info = self._load_split_info()
        self.patients = self._index_patients()

    def _load_split_info(self) -> Dict[str, Any]:
        try:
            with open(DATASET_SPLIT_PATH, 'r') as f:
                return json.load(f)
        except Exception:
            return {"patients": {"train": [], "validation": [], "test": []}, "slice_counts": {}}

    def _index_patients(self) -> Dict[str, Dict[str, Any]]:
        # Flatten split info to quickly lookup a patient's split
        patient_to_split = {}
        for split_name, patient_list in self.split_info.get("patients", {}).items():
            for pid in patient_list:
                patient_to_split[pid] = split_name

        CACHE_DATA_PATH = PROJECT_ROOT / "LiverCancer-MultiAgent-Retrieval" / "data" / "cache" / "cases"
        patients_data = {}
        if not CACHE_DATA_PATH.exists():
            return patients_data

        for p_dir in CACHE_DATA_PATH.iterdir():
            if not p_dir.is_dir():
                continue
            
            pid = p_dir.name
            
            # Since these are real generated cases, the patient ID is the first part of the UUID (e.g. 20dfdb3d19c8)
            # Or we can just use the whole case_id as pid
            
            image_file = p_dir / "image.npy"
            if image_file.exists():
                import numpy as np
                # Use mmap to quickly read shape without loading array
                try:
                    images = np.load(image_file, mmap_mode='r')
                    num_slices = len(images)
                except Exception:
                    num_slices = 0
            else:
                num_slices = 0

            if num_slices > 0:
                patients_data[pid] = {
                    "patient_id": pid,
                    "split": patient_to_split.get(pid.split('_')[0], "test"), # Guess split from original patient ID
                    "num_slices": num_slices,
                    "classes_present": ["Background", "Liver", "Mass", "Portal vein", "Abdominal aorta"]
                }
                
        return patients_data

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_patients": len(self.patients),
            "split_distribution": self.split_info.get("slice_counts", {}),
            "classes": ["Background", "Liver", "Mass", "Portal vein", "Abdominal aorta"]
        }

    def get_all_patients(self) -> List[Dict[str, Any]]:
        # Sort by patient ID
        return sorted(list(self.patients.values()), key=lambda x: x["patient_id"])

    def get_patient(self, patient_id: str) -> Dict[str, Any]:
        return self.patients.get(patient_id)

dataset_service = DatasetService()
