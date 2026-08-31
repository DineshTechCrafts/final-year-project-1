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

        patients_data = {}
        if not PROCESSED_DATA_PATH.exists():
            return patients_data

        for p_dir in PROCESSED_DATA_PATH.iterdir():
            if not p_dir.is_dir():
                continue
            
            pid = p_dir.name
            
            # Normalize to hcc_XXX if needed (the prompt says to handle hcc_055 canonical)
            if pid == "hcc055":
                continue # Skip the deleted one if it somehow lingers

            images_dir = p_dir / "images"
            if images_dir.exists():
                slices = list(images_dir.glob("*.png"))
                num_slices = len(slices)
            else:
                num_slices = 0

            # Only include patients with valid slices
            if num_slices > 0:
                patients_data[pid] = {
                    "patient_id": pid,
                    "split": patient_to_split.get(pid, "unknown"),
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
