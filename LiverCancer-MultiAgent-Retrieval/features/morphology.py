import numpy as np
from skimage.measure import label, regionprops

def extract_morphology(mask: np.ndarray, class_idx: int) -> dict:
    """
    Extracts morphology features for a given class from a 2D mask.
    Returns explicitly 0 or None based on missing-value strategy.
    """
    binary_mask = (mask == class_idx).astype(np.uint8)
    
    if np.sum(binary_mask) == 0:
        return {
            f"area_px": 0,
            f"perimeter": 0.0,
            f"bbox_width": 0,
            f"bbox_height": 0,
            f"bbox_aspect_ratio": 0.0,
            f"equivalent_diameter": 0.0,
            f"centroid_x": np.nan,  # Technical missingness for position if absent
            f"centroid_y": np.nan,
            f"eccentricity": np.nan,
            f"solidity": np.nan,
            f"extent": 0.0,
            f"compactness": np.nan,
            f"circularity": np.nan,
            f"component_count": 0
        }
        
    labeled = label(binary_mask)
    props = regionprops(labeled)
    
    # We aggregate if there are multiple components (e.g. for Mass)
    # The centroid will be the center of mass of the largest component, or globally
    # We calculate global area.
    
    total_area = sum([p.area for p in props])
    largest_prop = max(props, key=lambda x: x.area)
    
    # Bounding box of the largest
    min_row, min_col, max_row, max_col = largest_prop.bbox
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col
    
    # Calculate perimeter
    total_perimeter = sum([p.perimeter for p in props])
    
    # Aspect ratio
    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
    
    circularity = (4 * np.pi * total_area) / (total_perimeter ** 2) if total_perimeter > 0 else 0
    
    return {
        f"area_px": total_area,
        f"perimeter": total_perimeter,
        f"bbox_width": bbox_width,
        f"bbox_height": bbox_height,
        f"bbox_aspect_ratio": aspect_ratio,
        f"equivalent_diameter": largest_prop.equivalent_diameter_area,
        f"centroid_x": largest_prop.centroid[1],
        f"centroid_y": largest_prop.centroid[0],
        f"eccentricity": largest_prop.eccentricity,
        f"solidity": largest_prop.solidity,
        f"extent": largest_prop.extent,
        f"compactness": total_perimeter ** 2 / (total_area + 1e-5),
        f"circularity": circularity,
        f"component_count": len(props)
    }
