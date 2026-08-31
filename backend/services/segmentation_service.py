from pathlib import Path
from fastapi.responses import FileResponse, Response
from fastapi import HTTPException
import numpy as np
from PIL import Image
from io import BytesIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

class SegmentationService:
    def __init__(self):
        # 0 = Background (Transparent)
        # 1 = Liver (Red)
        # 2 = Mass (Yellow)
        # 3 = Portal vein (Blue)
        # 4 = Abdominal aorta (Green)
        self.color_map = {
            0: [0, 0, 0, 0],           # Transparent
            1: [255, 50, 50, 100],     # Liver - Red/Pink
            2: [255, 255, 0, 180],     # Mass - Yellow (More opaque)
            3: [50, 50, 255, 150],     # Portal Vein - Blue
            4: [50, 255, 50, 150]      # Aorta - Green
        }

    def get_slice_mask_path(self, patient_id: str, slice_index: int) -> Path:
        filename = f"{slice_index:04d}.png"
        mask_path = PROCESSED_DATA_PATH / patient_id / "masks" / filename
        
        if not mask_path.exists():
            raise HTTPException(status_code=404, detail="Mask slice not found")
            
        return mask_path

    def get_colored_mask_png(self, patient_id: str, slice_index: int, active_classes: list[int]) -> bytes:
        mask_path = self.get_slice_mask_path(patient_id, slice_index)
        mask_img = Image.open(mask_path).convert('L')
        mask_arr = np.array(mask_img)
        
        # Create an RGBA image
        rgba_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
        
        for class_idx, color in self.color_map.items():
            if class_idx == 0:
                continue
            if class_idx in active_classes:
                # Apply color where mask equals class_idx
                rgba_arr[mask_arr == class_idx] = color
                
        out_img = Image.fromarray(rgba_arr, 'RGBA')
        buf = BytesIO()
        out_img.save(buf, format='PNG')
        return buf.getvalue()

segmentation_service = SegmentationService()
