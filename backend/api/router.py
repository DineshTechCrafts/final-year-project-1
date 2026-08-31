from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse, Response
from backend.services.dataset_service import dataset_service
from backend.services.image_service import image_service
from backend.services.segmentation_service import segmentation_service
from backend.services.retrieval_service import retrieval_service

api_router = APIRouter()

@api_router.get("/health")
def get_health():
    return {"status": "ok"}

@api_router.get("/stats")
def get_stats():
    return dataset_service.get_stats()

@api_router.get("/patients")
def get_patients():
    return {"patients": dataset_service.get_all_patients()}

@api_router.get("/patients/{patient_id}")
def get_patient(patient_id: str):
    p = dataset_service.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p

@api_router.get("/patients/{patient_id}/slices")
def get_patient_slices(patient_id: str):
    p = dataset_service.get_patient(patient_id)
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"total_slices": p["num_slices"]}

@api_router.get("/patients/{patient_id}/slice/{slice_index}/image")
def get_slice_image(patient_id: str, slice_index: int):
    path = image_service.get_slice_image_path(patient_id, slice_index)
    return FileResponse(path, media_type="image/png")

@api_router.get("/patients/{patient_id}/slice/{slice_index}/mask")
def get_slice_mask(patient_id: str, slice_index: int):
    path = segmentation_service.get_slice_mask_path(patient_id, slice_index)
    return FileResponse(path, media_type="image/png")

@api_router.get("/patients/{patient_id}/slice/{slice_index}/overlay")
def get_slice_overlay(
    patient_id: str, 
    slice_index: int,
    classes: str = Query("1,2,3,4", description="Comma separated list of classes to include")
):
    try:
        active_classes = [int(c) for c in classes.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid classes parameter")
        
    img_bytes = segmentation_service.get_colored_mask_png(patient_id, slice_index, active_classes)
    return Response(content=img_bytes, media_type="image/png")

@api_router.get("/retrieval/similar/{patient_id}")
def get_similar_cases(patient_id: str, slice_index: int = None):
    return retrieval_service.get_similar_cases(patient_id, slice_index)

@api_router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    result = retrieval_service.process_uploaded_image(contents)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

@api_router.get("/model/status")
def get_model_status():
    return {
        "Dataset": {"status": "Ready", "color": "green"},
        "Preprocessing": {"status": "Complete", "color": "green"},
        "Segmentation": {"status": "Available (Ground Truth)", "color": "blue"},
        "Retrieval Engine": {"status": "Demo Mode", "color": "yellow"},
        "Embedding Model": {"status": "Pending Training", "color": "gray"},
        "Multi-Agent Pipeline": {"status": "Prototype", "color": "yellow"}
    }
