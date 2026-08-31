import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_file_hash(filepath: Path) -> str:
    if not filepath.exists():
        return "MISSING"
    
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "ERROR_HASHING"

def audit_file(filepath: Path) -> dict:
    exists = filepath.exists()
    return {
        "path": str(filepath.relative_to(PROJECT_ROOT)) if exists else str(filepath),
        "exists": exists,
        "hash": generate_file_hash(filepath) if exists else "MISSING"
    }

def main():
    print("Auditing dependencies for Stage 12 Inference Engine...")
    
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        "status": "STAGE12_AUDIT_COMPLETE",
        "directive": "The web application MUST reuse these exact artifacts. Do not retrain.",
        "dependencies": {
            "Stage_4_Segmentation": {
                "checkpoint": audit_file(PROJECT_ROOT / "models" / "segmentation_best.pth") # Mock path representation
            },
            "Stage_5_Features": {
                "scaler": audit_file(PROJECT_ROOT / "artifacts" / "feature_scaler.pkl"),
                "imputer": audit_file(PROJECT_ROOT / "artifacts" / "feature_imputer.pkl")
            },
            "Stage_6_Visual_Embedding": {
                "encoder": "ResNet50_ImageNet",
                "expected_dim": 2048
            },
            "Stage_7_Fusion": {
                "expected_dim": 256,
                "fusion_checkpoint": audit_file(PROJECT_ROOT / "models" / "gated_fusion.pth") # Mock path
            },
            "Stage_8_FAISS": {
                "gated_index": audit_file(PROJECT_ROOT / "data" / "vector_db" / "gated.index"),
                "gated_metadata": audit_file(PROJECT_ROOT / "data" / "vector_db" / "gated_metadata.parquet")
            },
            "Stage_9_Reranking": {
                "configuration": "Deterministic Multi-Agent + Gated Weights"
            },
            "Stage_10_Explanation": {
                "configuration": "Deterministic Template Generator"
            },
            "Stage_11_Frozen_Experiment": {
                "config": audit_file(PROJECT_ROOT / "configs" / "stage11_frozen_experiment.yaml")
            }
        }
    }
    
    out_path = metadata_dir / "stage12_dependency_manifest.json"
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=4)
        
    print(f"Audit complete. Manifest saved to: {out_path.relative_to(PROJECT_ROOT)}")

if __name__ == "__main__":
    main()
