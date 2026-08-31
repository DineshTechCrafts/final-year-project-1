import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyDiceLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=255, dice_weight=0.5):
        super().__init__()
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None
        
        self.ignore_index = ignore_index
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=self.class_weights, ignore_index=ignore_index)
        
    def forward(self, logits, targets):
        """
        logits: (B, C, H, W)
        targets: (B, H, W)
        """
        ce_loss = self.ce(logits, targets)
        
        # Dice Loss
        probs = F.softmax(logits, dim=1)
        num_classes = probs.shape[1]
        
        dice_loss = 0.0
        active_classes = 0
        
        # Create one-hot targets, ignoring ignore_index
        valid_mask = (targets != self.ignore_index)
        
        for c in range(num_classes):
            # We don't want to calculate dice for background if we don't want to, 
            # but usually it's fine. We will calculate dice for all classes and average.
            pred_c = probs[:, c, :, :]
            target_c = (targets == c).float()
            
            # Mask out ignore index
            pred_c = pred_c * valid_mask.float()
            
            intersection = torch.sum(pred_c * target_c)
            cardinality = torch.sum(pred_c) + torch.sum(target_c)
            
            if cardinality > 0:
                dice = (2.0 * intersection + 1e-5) / (cardinality + 1e-5)
                dice_loss += (1.0 - dice)
                active_classes += 1
                
        if active_classes > 0:
            dice_loss = dice_loss / active_classes
            
        return (1.0 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
