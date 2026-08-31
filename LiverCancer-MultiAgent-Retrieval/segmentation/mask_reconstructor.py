import numpy as np
import yaml
from pathlib import Path

class MaskReconstructor:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["colors"]
            
        self.colors = []
        self.labels = []
        for name, data in self.config.items():
            self.colors.append(data["rgb"])
            lbl = data["label"]
            if lbl == "UNKNOWN":
                lbl = 255
            self.labels.append(lbl)
            
        self.colors_np = np.array(self.colors, dtype=np.float32)
        self.labels_np = np.array(self.labels, dtype=np.uint8)

    def reconstruct(self, rgb_mask: np.ndarray, threshold: float = 40.0):
        """
        Reconstructs a discrete label mask and an uncertainty mask from an RGB JPEG mask.
        """
        H, W, C = rgb_mask.shape
        flat = rgb_mask.reshape(-1, 3).astype(np.float32)
        
        # Calculate Euclidean distances: shape (N, num_colors)
        distances = np.linalg.norm(flat[:, None, :] - self.colors_np[None, :, :], axis=2)
        
        # Find nearest color index and its distance
        min_indices = np.argmin(distances, axis=1)
        min_distances = np.min(distances, axis=1)
        
        # Assign configured labels
        reconstructed = self.labels_np[min_indices]
        
        # Mask ambiguous pixels
        uncertain = (min_distances > threshold)
        reconstructed[uncertain] = 255
        
        # Also mark as UNCERTAIN if the config label was UNKNOWN (255)
        # We don't want them in the confident mask.
        reconstructed_2d = reconstructed.reshape(H, W).astype(np.uint8)
        uncertainty_2d = (uncertain * 255).reshape(H, W).astype(np.uint8)
        
        return reconstructed_2d, uncertainty_2d
