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

    def get_slice_mask_path(self, case_id: str) -> Path:
        mask_path = PROJECT_ROOT / "LiverCancer-MultiAgent-Retrieval" / "data" / "cache" / "cases" / case_id / "mask.npy"
        
        if not mask_path.exists():
            raise HTTPException(status_code=404, detail="Mask slice not found")
            
        return mask_path

    def get_slice_mask_png(self, case_id: str, slice_index: int) -> bytes:
        from PIL import Image
        from io import BytesIO
        mask_path = self.get_slice_mask_path(case_id)
        masks = np.load(mask_path, mmap_mode='r')
        if slice_index >= len(masks) or slice_index < 0:
            raise HTTPException(status_code=404, detail="Slice index out of bounds")
            
        mask_arr = masks[slice_index]
        img_pil = Image.fromarray((mask_arr * 50).astype(np.uint8)).convert('L') # Multiply by 50 for visibility
        buf = BytesIO()
        img_pil.save(buf, format='PNG')
        return buf.getvalue()

    def get_colored_mask_png(self, case_id: str, slice_index: int, active_classes: list[int]) -> bytes:
        mask_path = self.get_slice_mask_path(case_id)
        image_path = PROJECT_ROOT / "LiverCancer-MultiAgent-Retrieval" / "data" / "cache" / "cases" / case_id / "image.npy"
        
        masks = np.load(mask_path, mmap_mode='r')
        if slice_index >= len(masks) or slice_index < 0:
            raise HTTPException(status_code=404, detail="Slice index out of bounds")
            
        mask_arr = masks[slice_index]
        
        # Load corresponding image slice
        images = np.load(image_path, mmap_mode='r')
        img_arr = images[slice_index]
        img_rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
            
        base_img = Image.fromarray(img_rgb).convert('RGBA')
        
        # Create an RGBA mask
        rgba_arr = np.zeros((*mask_arr.shape, 4), dtype=np.uint8)
        
        for class_idx, color in self.color_map.items():
            if class_idx == 0:
                continue
            if class_idx in active_classes:
                rgba_arr[mask_arr == class_idx] = color
                
        overlay_img = Image.fromarray(rgba_arr, 'RGBA')
        final_img = Image.alpha_composite(base_img, overlay_img)
        
        buf = BytesIO()
        final_img.save(buf, format='PNG')
        return buf.getvalue()

segmentation_service = SegmentationService()
