import os
import glob
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def generate_file_hash(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception:
        return "ERROR_HASHING"

def main():
    print("Discovering frozen model artifacts...")
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Search for actual model files
    extensions = ["*.pth", "*.pt", "*.ckpt", "*.bin", "*.safetensors"]
    discovered_files = []
    
    for ext in extensions:
        # Search recursively
        for filepath in PROJECT_ROOT.rglob(ext):
            # Exclude virtual environment
            if "venv" in filepath.parts:
                continue
                
            path_str = str(filepath.name).lower()
            role = "unknown"
            if "segment" in path_str or "unet" in path_str:
                role = "segmentation"
            elif "resnet" in path_str or "visual" in path_str:
                role = "visual_encoder"
            elif "fusion" in path_str or "projection" in path_str:
                role = "fusion"
                
            discovered_files.append({
                "path": str(filepath.relative_to(PROJECT_ROOT)),
                "size_bytes": filepath.stat().st_size,
                "sha256": generate_file_hash(filepath),
                "modified_time": str(filepath.stat().st_mtime),
                "possible_role": role
            })
            
    with open(metadata_dir / "stage13b_model_discovery.json", "w") as f:
        json.dump(discovered_files, f, indent=4)
        
    # 2. Search for configuration references
    references = []
    keywords = [
        "segmentation_best.pth", "best_validation_mass_dice.pt",
        "ResNet50", "ResNet34", "GatedFusion", "ProjectionNetworks",
        "torch.save", "torch.load", "state_dict", "checkpoint"
    ]
    
    for filepath in PROJECT_ROOT.rglob("*.py"):
        if "venv" in filepath.parts:
            continue
            
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    for kw in keywords:
                        if kw in line:
                            references.append({
                                "source_file": str(filepath.relative_to(PROJECT_ROOT)),
                                "line_number": i + 1,
                                "matched_keyword": kw,
                                "line_content": line.strip()
                            })
        except Exception:
            pass
            
    with open(metadata_dir / "stage13b_checkpoint_references.json", "w") as f:
        json.dump(references, f, indent=4)
        
    print("Discovery complete.")

if __name__ == "__main__":
    main()
