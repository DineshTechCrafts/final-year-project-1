import numpy as np
import cv2
from skimage.measure import label, regionprops

def extract_target_crop(image: np.ndarray, mask: np.ndarray, class_idx: int, margin: int = 15):
    """
    Finds bounding box of the target class in the mask.
    Expands by margin, clamps to image borders, and extracts the crop from the image.
    
    Returns:
        crop: np.ndarray (or None if missing)
        bbox: (min_row, min_col, max_row, max_col)
    """
    binary_mask = (mask == class_idx).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return None, None
        
    labeled = label(binary_mask)
    props = regionprops(labeled)
    
    # Bound across all components
    min_row = min([p.bbox[0] for p in props])
    min_col = min([p.bbox[1] for p in props])
    max_row = max([p.bbox[2] for p in props])
    max_col = max([p.bbox[3] for p in props])
    
    H, W = image.shape[:2]
    
    # Expand and clamp
    min_row = max(0, min_row - margin)
    min_col = max(0, min_col - margin)
    max_row = min(H, max_row + margin)
    max_col = min(W, max_col + margin)
    
    crop = image[min_row:max_row, min_col:max_col]
    return crop, (min_row, min_col, max_row, max_col)
