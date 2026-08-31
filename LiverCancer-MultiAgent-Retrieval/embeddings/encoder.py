import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VisualEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet50(weights=weights)
        
        # Remove the classification head, keep up to avgpool
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
    def forward(self, x, normalize=True):
        """
        x: (B, 3, H, W) tensor
        returns: (B, 2048) embedding
        """
        # Forward pass through resnet without FC
        features = self.encoder(x)
        # Flatten
        features = torch.flatten(features, 1)
        
        if normalize:
            # L2 normalization, safe handling for zero vectors
            features = F.normalize(features, p=2, dim=1, eps=1e-8)
            
        return features
