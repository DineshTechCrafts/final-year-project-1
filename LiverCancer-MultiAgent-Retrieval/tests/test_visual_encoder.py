import torch
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from embeddings.encoder import VisualEncoder

def test_visual_encoder_dimensions():
    # Model without pretrained weights for fast testing
    model = VisualEncoder(pretrained=False)
    model.eval()
    
    # Fake RGB tensor (B, C, H, W) -> (2, 3, 224, 224)
    x = torch.rand(2, 3, 224, 224)
    
    with torch.no_grad():
        embeddings = model(x, normalize=True)
        
    assert embeddings.shape == (2, 2048), "Encoder output dimension must be exactly 2048."
    
    # Check L2 norm is ~1
    norms = torch.norm(embeddings, p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2)), "Embeddings must be L2 normalized."
    
def test_encoder_handles_zero_safely():
    model = VisualEncoder(pretrained=False)
    model.eval()
    
    x = torch.zeros(1, 3, 224, 224)
    
    with torch.no_grad():
        embeddings = model(x, normalize=True)
        
    assert not torch.isnan(embeddings).any(), "Safe normalization failed on zero input."
    assert not torch.isinf(embeddings).any(), "Safe normalization failed on zero input."
