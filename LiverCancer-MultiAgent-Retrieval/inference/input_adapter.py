import hashlib
from pathlib import Path

class InputAdapter:
    def validate(self, filepath: str) -> dict:
        path = Path(filepath)
        if not path.exists():
            return {"status": "error", "message": "File does not exist."}
            
        # Basic validation (in a real scenario, use PIL to verify dimensions, readability)
        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            file_hash = sha256.hexdigest()
        except Exception:
            return {"status": "error", "message": "File unreadable."}
            
        return {
            "status": "success",
            "file_hash": file_hash,
            "filename": path.name,
            "validated": True
        }
