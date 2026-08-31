import torch
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.fusion_models import MultimodalFusionEngine
from fusion.losses import InfoNCELoss

def test_multimodal_fusion_shapes():
    # 45 numerical structured features, 2048 visual embeddings
    model = MultimodalFusionEngine(struct_in_dim=45, vis_in_dim=2048, hidden_dim=512, out_dim=256)
    
    B = 8
    struct_x = torch.randn(B, 45)
    vis_x = torch.randn(B, 2048)
    
    outputs = model(struct_x, vis_x)
    
    assert outputs["structured"].shape == (B, 256)
    assert outputs["visual"].shape == (B, 256)
    assert outputs["concat"].shape == (B, 256)
    assert outputs["weighted"].shape == (B, 256)
    assert outputs["gated"].shape == (B, 256)
    assert outputs["gate_values"].shape == (B, 256)
    
    # Check L2 norms
    norm = torch.norm(outputs["gated"], p=2, dim=1)
    assert torch.allclose(norm, torch.ones(B)), "Outputs must be L2 normalized."

def test_infonce_loss():
    loss_fn = InfoNCELoss(temperature=0.1)
    z_i = torch.randn(4, 256)
    z_j = torch.randn(4, 256)
    
    z_i = torch.nn.functional.normalize(z_i, p=2, dim=1)
    z_j = torch.nn.functional.normalize(z_j, p=2, dim=1)
    
    loss = loss_fn(z_i, z_j)
    
    assert loss.dim() == 0, "Loss must be scalar."
    assert not torch.isnan(loss), "Loss cannot be NaN."
