import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_file_hash(filepath: Path) -> str:
    if not filepath.exists():
        return "MISSING_FROZEN_ARTIFACT"
    
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "ERROR_HASHING"

def inventory_item(filepath: Path, artifact_type: str, required: bool = True) -> dict:
    exists = filepath.exists()
    return {
        "path": str(filepath.relative_to(PROJECT_ROOT)) if filepath.is_absolute() else str(filepath),
        "exists": exists,
        "sha256": generate_file_hash(filepath),
        "artifact_type": artifact_type,
        "required_for_inference": required
    }

def main():
    print("Inventorying frozen artifacts for Stage 13A Inference...")
    
    inventory = {
        "segmentation_model": inventory_item(PROJECT_ROOT / "models" / "segmentation_best.pth", "pytorch_checkpoint"),
        "feature_scaler": inventory_item(PROJECT_ROOT / "artifacts" / "feature_scaler.pkl", "sklearn_scaler"),
        "feature_imputer": inventory_item(PROJECT_ROOT / "artifacts" / "feature_imputer.pkl", "sklearn_imputer"),
        "fusion_model": inventory_item(PROJECT_ROOT / "models" / "gated_fusion.pth", "pytorch_checkpoint"),
        "faiss_index": inventory_item(PROJECT_ROOT / "data" / "vector_db" / "gated.index", "faiss_binary"),
        "faiss_metadata": inventory_item(PROJECT_ROOT / "data" / "vector_db" / "gated_metadata.parquet", "parquet_dataframe"),
        "stage11_config": inventory_item(PROJECT_ROOT / "configs" / "stage11_frozen_experiment.yaml", "yaml_config")
    }
    
    out_path = PROJECT_ROOT / "data" / "metadata" / "stage13a_artifact_inventory.json"
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=4)
        
    print(f"Inventory complete. Saved to: {out_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
