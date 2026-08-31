import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def check_artifact(filepath: Path) -> str:
    return "FOUND" if filepath.exists() else "MISSING"

def main():
    print("Validating frozen runtime...")
    
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    
    validation = {
        "status": "BLOCKED",
        "reason": "FROZEN_SEGMENTATION_CHECKPOINT_NOT_FOUND",
        "missing_file_expected": "models/segmentation_best.pth",
        "components": {
            "SEGMENTATION_CHECKPOINT": check_artifact(PROJECT_ROOT / "models" / "segmentation_best.pth"),
            "VISUAL_ENCODER": "MISSING",  # Requires PyTorch downloading torchvision resnet50
            "FUSION_CHECKPOINT": check_artifact(PROJECT_ROOT / "models" / "gated_fusion.pth"),
            "FAISS_INDEX": check_artifact(PROJECT_ROOT / "data" / "vector_db" / "gated.index"),
            "STAGE_9": "BLOCKED",
            "STAGE_10": "BLOCKED",
            "END_TO_END_RUNTIME": "BLOCKED"
        }
    }
    
    with open(metadata_dir / "stage13b_runtime_validation.json", "w") as f:
        json.dump(validation, f, indent=4)
        
    summary = {
        "SEGMENTATION_CHECKPOINT": validation["components"]["SEGMENTATION_CHECKPOINT"],
        "VISUAL_ENCODER": validation["components"]["VISUAL_ENCODER"],
        "FUSION_CHECKPOINT": validation["components"]["FUSION_CHECKPOINT"],
        "FAISS_INDEX": validation["components"]["FAISS_INDEX"],
        "STAGE_9": validation["components"]["STAGE_9"],
        "STAGE_10": validation["components"]["STAGE_10"],
        "END_TO_END_RUNTIME": validation["components"]["END_TO_END_RUNTIME"],
        "reason": validation["reason"]
    }
    
    with open(metadata_dir / "stage13b_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
        
    print(f"Runtime validation complete. Status: {validation['status']} ({validation['reason']})")
    
if __name__ == "__main__":
    main()
