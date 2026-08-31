import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCELoss(nn.Module):
    """
    Self-supervised contrastive loss.
    Assumes batch is formulated as pairs of (anchor, positive) adjacent in the batch or structurally defined.
    If features are shaped (B, D) and represent anchors, and we have a corresponding (B, D) of positives.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        """
        z_i, z_j: (B, D) representations, L2 normalized.
        z_i[k] and z_j[k] are a positive pair.
        """
        batch_size = z_i.size(0)
        
        # (2B, D)
        z = torch.cat([z_i, z_j], dim=0)
        
        # similarity matrix (2B, 2B)
        sim = torch.matmul(z, z.T) / self.temperature
        
        # mask out self-similarity
        mask = torch.eye(2 * batch_size, device=sim.device).bool()
        sim.masked_fill_(mask, -9e15)
        
        # The positive for i in [0, B-1] is i + B
        # The positive for i in [B, 2B-1] is i - B
        pos_labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=sim.device),
            torch.arange(0, batch_size, device=sim.device)
        ], dim=0)
        
        loss = F.cross_entropy(sim, pos_labels)
        return loss
