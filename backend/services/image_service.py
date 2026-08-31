from pathlib import Path
from fastapi.responses import FileResponse
from fastapi import HTTPException
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

class ImageService:
    def get_slice_image_path(self, patient_id: str, slice_index: int) -> Path:
        # Format index to 4 digits: 0000.png
        filename = f"{slice_index:04d}.png"
        image_path = PROCESSED_DATA_PATH / patient_id / "images" / filename
        
        if not image_path.exists():
            raise HTTPException(status_code=404, detail="Image slice not found")
            
        return image_path

image_service = ImageService()
