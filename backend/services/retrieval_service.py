import json
import math
import sys
from typing import Dict, Any, List
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import io
import torch
import segmentation_models_pytorch as smp
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ML_ROOT = PROJECT_ROOT / "LiverCancer-MultiAgent-Retrieval"

if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from segmentation.transforms import get_transforms
from embeddings.encoder import VisualEncoder
from fusion.fusion_models import MultimodalFusionEngine
from retrieval.retriever import CaseRetriever
import faiss
from scripts.execute_stage5 import extract_slice_features_dict
from embeddings.crop import extract_target_crop

# Copying preprocess_crop_to_tensor from stage 6 for visual encoder
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

def preprocess_crop_to_tensor(crop: np.ndarray, target_size: tuple = (224, 224)) -> torch.Tensor:
    if len(crop.shape) == 2:
        crop_3ch = np.stack([crop, crop, crop], axis=-1)
    else:
        crop_3ch = crop
    resized = cv2.resize(crop_3ch, target_size, interpolation=cv2.INTER_LINEAR)
    norm = (resized.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm).permute(2, 0, 1).float()

class RetrievalService:
    def __init__(self):
        self.is_demo_mode = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Models
        self.seg_model = None
        self.vis_encoder = None
        self.fusion_model = None
        
        # Data
        self.struct_mean = None
        self.struct_std = None
        self.feat_cols = None
        self.retriever = None
        self.slice_retriever = None

    def _lazy_load_models(self):
        if self.seg_model is None:
            print("Lazy loading U-Net...")
            seg_path = ML_ROOT / "runs" / "segmentation_baseline" / "exp_C" / "best_validation_mass_dice.pt"
            self.seg_model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=5).to(self.device)
            self.seg_model.load_state_dict(torch.load(seg_path, map_location=self.device))
            self.seg_model.eval()

        if self.vis_encoder is None:
            print("Lazy loading VisualEncoder...")
            self.vis_encoder = VisualEncoder(pretrained=True).to(self.device)
            self.vis_encoder.eval()

        if self.fusion_model is None:
            print("Lazy loading GatedFusion Engine...")
            models_dir = ML_ROOT / "data" / "models"
            
            with open(models_dir / "feat_cols.json", "r") as f:
                self.feat_cols = json.load(f)
                
            scaler = np.load(models_dir / "struct_scaler.npz")
            self.struct_mean = scaler['mean']
            self.struct_std = scaler['std']
            
            self.fusion_model = MultimodalFusionEngine(struct_in_dim=len(self.feat_cols)).to(self.device)
            self.fusion_model.load_state_dict(torch.load(models_dir / "gated_fusion.pt", map_location=self.device))
            self.fusion_model.eval()

        if self.retriever is None:
            print("Loading FAISS index (case-level)...")
            vector_db_dir = ML_ROOT / "data" / "vector_db"
            index = faiss.read_index(str(vector_db_dir / "gated.index"))
            metadata = pd.read_parquet(vector_db_dir / "gated_metadata.parquet")
            self.retriever = CaseRetriever(index, metadata)
            
        if self.slice_retriever is None:
            print("Loading FAISS index (slice-level)...")
            vector_db_dir = ML_ROOT / "data" / "vector_db"
            s_index = faiss.read_index(str(vector_db_dir / "gated_slice.index"))
            s_metadata = pd.read_parquet(vector_db_dir / "gated_slice_metadata.parquet")
            self.slice_retriever = CaseRetriever(s_index, s_metadata)

    def get_similar_cases(self, query_patient_id: str, query_slice_index: int) -> Dict[str, Any]:
        """
        Retrieves similar cases based on an existing query case.
        """
        try:
            self._lazy_load_models()
        except Exception as e:
            return {"error": f"Failed to load models: {str(e)}"}
            
        try:
            # Note: For existing cases, search_by_case_id automatically fetches the fused case-embedding!
            # If query_patient_id is something like "hcc_001", we need to map to case_id if they are the same.
            # In our dataset, case_id == patient_id for training sets, or it's formatted. 
            # We will use query_patient_id as the case_id.
            results = self.retriever.search_by_case_id(query_patient_id, top_k=3, exclude_same_patient=True)
            return {
                "query_case_id": query_patient_id,
                "demo_mode": False,
                "retrieval_mode": "real_faiss_index",
                "results": results
            }
        except Exception as e:
            return {"error": f"Failed to retrieve cases: {str(e)}"}

    def process_uploaded_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        End-to-End inference on a single uploaded slice.
        """
        try:
            self._lazy_load_models()
            
            # 1. Base Image Parsing
            img = Image.open(io.BytesIO(image_bytes)).convert('L')
            img_arr = np.array(img)
            img_3c = np.stack([img_arr, img_arr, img_arr], axis=-1)
            
            # 2. Segmentation
            transforms = get_transforms(is_train=False)
            tensor = transforms(image=img_3c)["image"].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.seg_model(tensor)
                mask_arr = output.argmax(dim=1).squeeze(0).cpu().numpy()
                
            # --- DEBUG OVERLAY ---
            from backend.services.segmentation_service import segmentation_service
            rgba_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
            for class_idx, color in segmentation_service.color_map.items():
                if class_idx == 0: continue
                rgba_arr[mask_arr == class_idx] = color
            out_img = Image.alpha_composite(Image.fromarray(img_3c).convert("RGBA"), Image.fromarray(rgba_arr, "RGBA"))
            out_img.save(r"c:\Users\DINESH\.gemini\antigravity-ide\brain\fa52bb67-8726-4f59-bc7d-98402988b2ab\scratch\debug_overlay.png")
            # ---------------------
                
            # 3. Structured Features
            feat_dict = extract_slice_features_dict(img_arr, mask_arr, "upload", "upload", 0, "upload")
            # Extract exactly the columns needed, replace None/NaN with 0.0
            raw_struct = []
            for c in self.feat_cols:
                val = feat_dict.get(c, 0.0)
                if val is None or np.isnan(val):
                    val = 0.0
                raw_struct.append(val)
                
            struct_vector = np.array([raw_struct], dtype=np.float32)
            struct_norm = (struct_vector - self.struct_mean) / self.struct_std
            struct_tensor = torch.from_numpy(struct_norm).to(self.device)
            
            # 4. Visual Features (Tumor Crop)
            tumor_crop, _ = extract_target_crop(img_arr, mask_arr, class_idx=2)
            if tumor_crop is None:
                # Fallback to liver if no tumor
                tumor_crop, _ = extract_target_crop(img_arr, mask_arr, class_idx=1)
            if tumor_crop is None:
                # Fallback to full image
                tumor_crop = img_arr
                
            vis_tensor = preprocess_crop_to_tensor(tumor_crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                vis_emb = self.vis_encoder(vis_tensor, normalize=True)
            
            # 5. Gated Fusion
            with torch.no_grad():
                fusion_outputs = self.fusion_model(struct_tensor, vis_emb)
                query_emb = fusion_outputs["gated"].cpu().numpy()
                
            # 6. FAISS Retrieval (Slice-Level)
            results = self.slice_retriever.search_by_embedding(query_emb, top_k=3, exclude_query_case_id="upload", exclude_patient_id="upload")
            
            # Since the slice retriever retrieves slice-level entries, the returned 'case_id' 
            # is actually the case_id, but the metadata should contain 'slice_index' if we want to show it.
            # Wait, does gated_slice_metadata.parquet contain 'slice_index'?
            # Yes, slice_features_ground_truth.parquet has 'slice_index'.
            # Let's ensure the frontend knows which slice was retrieved.
            
            # Also compute basic ratios for the UI display
            total_pixels = mask_arr.size
            liver_ratio = float((mask_arr == 1).sum() / total_pixels)
            mass_ratio = float((mask_arr == 2).sum() / total_pixels)
            
            return {
                "query_case_id": "upload",
                "demo_mode": False,
                "retrieval_mode": "real_faiss_index",
                "simulated_features": {
                    "mean_intensity": round(float(np.mean(img_arr)), 2),
                    "liver_ratio": round(liver_ratio, 4),
                    "mass_ratio": round(mass_ratio, 4)
                },
                "results": results
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Failed to process image: {str(e)}"}

retrieval_service = RetrievalService()
