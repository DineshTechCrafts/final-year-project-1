import json
import math
import random
from typing import Dict, Any, List
from pathlib import Path
from PIL import Image
import numpy as np
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = PROJECT_ROOT / "backend" / "demo_index.json"

class RetrievalService:
    def __init__(self):
        self.is_demo_mode = True
        self.index = []
        self._load_index()

    def _load_index(self):
        if INDEX_PATH.exists():
            with open(INDEX_PATH, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = []

    def _calculate_similarity(self, f1: Dict[str, float], f2: Dict[str, float]) -> float:
        # Simple weighted Euclidean distance converted to similarity score
        # Normalize features roughly (mean is ~0-255, ratio is 0-1)
        diff_mean = (f1["mean_intensity"] - f2["mean_intensity"]) / 255.0
        diff_liver = f1["liver_ratio"] - f2["liver_ratio"]
        diff_mass = f1["mass_ratio"] - f2["mass_ratio"]
        diff_portal = f1["portal_ratio"] - f2["portal_ratio"]
        
        # Give higher weight to mass and liver structure
        dist = math.sqrt(
            1.0 * (diff_mean ** 2) +
            2.0 * (diff_liver ** 2) +
            3.0 * (diff_mass ** 2) +
            1.0 * (diff_portal ** 2)
        )
        # Convert distance to similarity 0-1
        sim = 1.0 / (1.0 + dist)
        return sim

    def get_similar_cases(self, query_patient_id: str, query_slice_index: int) -> Dict[str, Any]:
        """
        Retrieves similar cases based on the query.
        Uses deterministic mock features based on mask statistics for realistic demonstration.
        """
        # Reload if empty
        if not self.index:
            self._load_index()
            
        if not self.index:
            return {
                "query_case_id": query_patient_id,
                "demo_mode": self.is_demo_mode,
                "results": []
            }

        # Find query features
        query_entry = next((e for e in self.index if e["patient_id"] == query_patient_id and e["slice_index"] == query_slice_index), None)
        
        if not query_entry:
            return {
                "query_case_id": query_patient_id,
                "demo_mode": self.is_demo_mode,
                "error": "Query slice not found in feature index",
                "results": []
            }

        query_features = query_entry["features"]
        
        results = []
        seen_patients = set()
        
        # Calculate similarity with all other slices
        scored_candidates = []
        for entry in self.index:
            if entry["patient_id"] == query_patient_id:
                continue
                
            sim = self._calculate_similarity(query_features, entry["features"])
            scored_candidates.append((sim, entry))
            
        # Sort by highest similarity
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Pick top 3 from distinct patients
        for sim, entry in scored_candidates:
            if entry["patient_id"] not in seen_patients:
                seen_patients.add(entry["patient_id"])
                results.append({
                    "candidate_case_id": entry["patient_id"],
                    "similarity": round(sim, 3),
                    "matched_slice": entry["slice_index"]
                })
                if len(results) >= 3:
                    break

        return {
            "query_case_id": query_patient_id,
            "demo_mode": self.is_demo_mode,
            "results": results
        }

    def process_uploaded_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Takes an uploaded image and simulates real-time inference (segmentation -> retrieval).
        """
        try:
            # 1. Read real image stats
            img = Image.open(io.BytesIO(image_bytes)).convert('L')
            img_arr = np.array(img)
            mean_intensity = float(np.mean(img_arr))
        except Exception as e:
            return {"error": f"Failed to process image: {str(e)}"}
            
        # 2. Generate mock segmentation ratios (since U-Net isn't connected)
        # Typical HCC case features: liver ~5-10%, mass ~0.1-2%, portal ~0.2%, aorta ~0.1%
        liver_ratio = random.uniform(0.04, 0.15)
        mass_ratio = random.uniform(0.001, 0.03)
        portal_ratio = random.uniform(0.001, 0.005)
        aorta_ratio = random.uniform(0.0005, 0.002)
        
        mock_features = {
            "mean_intensity": mean_intensity,
            "liver_ratio": liver_ratio,
            "mass_ratio": mass_ratio,
            "portal_ratio": portal_ratio,
            "aorta_ratio": aorta_ratio
        }
        
        # 3. Retrieve similar cases
        if not self.index:
            self._load_index()
            
        scored_candidates = []
        for entry in self.index:
            sim = self._calculate_similarity(mock_features, entry["features"])
            scored_candidates.append((sim, entry))
            
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        seen_patients = set()
        
        for sim, entry in scored_candidates:
            if entry["patient_id"] not in seen_patients:
                seen_patients.add(entry["patient_id"])
                results.append({
                    "candidate_case_id": entry["patient_id"],
                    "similarity": round(sim, 3),
                    "matched_slice": entry["slice_index"]
                })
                if len(results) >= 3:
                    break
                    
        return {
            "query_case_id": "upload",
            "demo_mode": self.is_demo_mode,
            "simulated_features": {
                "mean_intensity": round(mean_intensity, 2),
                "liver_ratio": round(liver_ratio, 4),
                "mass_ratio": round(mass_ratio, 4)
            },
            "results": results
        }

retrieval_service = RetrievalService()
