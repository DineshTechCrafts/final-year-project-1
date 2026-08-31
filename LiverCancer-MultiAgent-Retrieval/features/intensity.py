import numpy as np
import scipy.ndimage as ndi

def extract_intensity(image: np.ndarray, mask: np.ndarray, class_idx: int) -> dict:
    """
    Extracts intensity features for a given class from the 8-bit image array.
    """
    binary_mask = (mask == class_idx)
    
    if np.sum(binary_mask) == 0:
        return {
            "mean": np.nan,
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p10": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p90": np.nan
        }
        
    pixels = image[binary_mask].astype(np.float32)
    
    return {
        "mean": float(np.mean(pixels)),
        "median": float(np.median(pixels)),
        "std": float(np.std(pixels)),
        "min": float(np.min(pixels)),
        "max": float(np.max(pixels)),
        "p10": float(np.percentile(pixels, 10)),
        "p25": float(np.percentile(pixels, 25)),
        "p75": float(np.percentile(pixels, 75)),
        "p90": float(np.percentile(pixels, 90))
    }

def extract_peritumoral_intensity(image: np.ndarray, mask: np.ndarray, tumor_class: int=2, radius: int=5) -> dict:
    """
    Extracts intensity features for the peritumoral ring (dilated tumor mask - original tumor mask).
    """
    tumor_mask = (mask == tumor_class)
    
    if np.sum(tumor_mask) == 0:
        return {
            "peritumoral_mean": np.nan,
            "peritumoral_median": np.nan,
            "peritumoral_std": np.nan
        }
        
    # Dilate
    struct_elem = ndi.generate_binary_structure(2, 1)
    dilated = ndi.binary_dilation(tumor_mask, structure=struct_elem, iterations=radius)
    ring_mask = dilated ^ tumor_mask
    
    # Restrict ring to background or liver (do not include other structures like aorta/vein if possible, but 
    # to be simple, we just restrict to liver + background if we want, or just anywhere). Let's restrict it to non-tumor.
    
    if np.sum(ring_mask) == 0:
         return {
            "peritumoral_mean": np.nan,
            "peritumoral_median": np.nan,
            "peritumoral_std": np.nan
        }
        
    pixels = image[ring_mask].astype(np.float32)
    return {
        "peritumoral_mean": float(np.mean(pixels)),
        "peritumoral_median": float(np.median(pixels)),
        "peritumoral_std": float(np.std(pixels))
    }
