from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import HTTPException
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

class ImageService:
    def get_slice_image_path(self, case_id: str) -> Path:
        image_path = PROJECT_ROOT / "LiverCancer-MultiAgent-Retrieval" / "data" / "cache" / "cases" / case_id / "image.npy"
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image volume not found")
            
        return image_path
        
    def get_slice_image_png(self, case_id: str, slice_index: int) -> bytes:
        image_path = self.get_slice_image_path(case_id)
        import numpy as np
        from PIL import Image
        from io import BytesIO
        
        images = np.load(image_path, mmap_mode='r')
        if slice_index >= len(images) or slice_index < 0:
            raise HTTPException(status_code=404, detail="Slice index out of bounds")
            
        img_arr = images[slice_index]
        img_rgb = np.stack([img_arr, img_arr, img_arr], axis=-1)
        
        if img_rgb.max() <= 1.0:
            img_rgb = (img_rgb * 255).astype(np.uint8)
        else:
            img_rgb = img_rgb.astype(np.uint8)
            
        img_pil = Image.fromarray(img_rgb).convert('L')
        buf = BytesIO()
        img_pil.save(buf, format='PNG')
        return buf.getvalue()

image_service = ImageService()
