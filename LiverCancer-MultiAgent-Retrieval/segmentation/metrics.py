import numpy as np

def calculate_metrics(preds, targets, num_classes=5, ignore_index=255):
    """
    preds: numpy array of shape (N, H, W)
    targets: numpy array of shape (N, H, W)
    """
    metrics = {
        "per_class": {}
    }
    
    valid_mask = (targets != ignore_index)
    preds = preds[valid_mask]
    targets = targets[valid_mask]
    
    mean_dice = 0.0
    mean_iou = 0.0
    active_classes = 0
    
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        
        intersection = np.sum(pred_c & target_c)
        union = np.sum(pred_c | target_c)
        
        if np.sum(target_c) > 0:
            dice = 2.0 * intersection / (np.sum(pred_c) + np.sum(target_c) + 1e-8)
            iou = intersection / (union + 1e-8)
            precision = intersection / (np.sum(pred_c) + 1e-8)
            recall = intersection / (np.sum(target_c) + 1e-8)
            
            metrics["per_class"][str(c)] = {
                "dice": float(dice),
                "iou": float(iou),
                "precision": float(precision),
                "recall": float(recall)
            }
            
            mean_dice += dice
            mean_iou += iou
            active_classes += 1
            
    if active_classes > 0:
        metrics["mean_dice"] = float(mean_dice / active_classes)
        metrics["mean_iou"] = float(mean_iou / active_classes)
        
    return metrics
