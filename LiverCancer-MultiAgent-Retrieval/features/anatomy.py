import numpy as np

def extract_anatomy(morph_liver: dict, morph_tumor: dict, morph_portal: dict, morph_aorta: dict, image_shape: tuple) -> dict:
    """
    Computes relative spatial and anatomical features.
    Handles NaN mathematically (representing absence).
    """
    features = {}
    
    H, W = image_shape
    
    # 1. Tumor to Liver Ratio
    if morph_liver["area_px"] > 0:
        features["tumor_to_liver_area_ratio"] = morph_tumor["area_px"] / morph_liver["area_px"]
    else:
        features["tumor_to_liver_area_ratio"] = np.nan
        
    # 2. Normalized centroids
    if morph_liver["area_px"] > 0 and morph_tumor["area_px"] > 0:
        # relative distance normalized by image size
        dx = (morph_tumor["centroid_x"] - morph_liver["centroid_x"]) / W
        dy = (morph_tumor["centroid_y"] - morph_liver["centroid_y"]) / H
        features["tumor_to_liver_centroid_distance"] = np.sqrt(dx**2 + dy**2)
    else:
        features["tumor_to_liver_centroid_distance"] = np.nan
        
    if morph_portal["area_px"] > 0 and morph_tumor["area_px"] > 0:
        dx = (morph_tumor["centroid_x"] - morph_portal["centroid_x"]) / W
        dy = (morph_tumor["centroid_y"] - morph_portal["centroid_y"]) / H
        features["tumor_to_portal_centroid_distance"] = np.sqrt(dx**2 + dy**2)
    else:
        features["tumor_to_portal_centroid_distance"] = np.nan
        
    if morph_aorta["area_px"] > 0 and morph_tumor["area_px"] > 0:
        dx = (morph_tumor["centroid_x"] - morph_aorta["centroid_x"]) / W
        dy = (morph_tumor["centroid_y"] - morph_aorta["centroid_y"]) / H
        features["tumor_to_aorta_centroid_distance"] = np.sqrt(dx**2 + dy**2)
    else:
        features["tumor_to_aorta_centroid_distance"] = np.nan
        
    return features
