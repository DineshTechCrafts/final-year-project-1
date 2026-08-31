import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from inference.pipeline import run_inference
from pathlib import Path

router = APIRouter()

UPLOAD_DIR = Path("tmp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/inference")
async def perform_inference(
    modality: str = Form(...),
    file: UploadFile = File(...)
):
    # Security: allowed file extensions
    allowed = [".jpg", ".jpeg", ".png"]
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/PNG supported.")
        
    # Temporary storage
    tmp_path = UPLOAD_DIR / f"{file.filename}"
    try:
        with open(tmp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result = run_inference(str(tmp_path), modality)
        
        if result["status"] == "ERROR":
            raise HTTPException(status_code=400, detail=result["message"])
            
        return result
        
    finally:
        # Automatic temporary-file cleanup
        if tmp_path.exists():
            os.remove(tmp_path)
