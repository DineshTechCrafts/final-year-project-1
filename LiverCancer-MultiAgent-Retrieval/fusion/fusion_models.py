import torch
import torch.nn as nn
import torch.nn.functional as F

class ProjectionNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )
        
    def forward(self, x):
        return self.net(x)

class ConcatFusion(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(emb_dim * 2, emb_dim),
            nn.LayerNorm(emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim)
        )
        
    def forward(self, struct_emb, vis_emb):
        fused = torch.cat([struct_emb, vis_emb], dim=1)
        return self.proj(fused)

class WeightedFusion(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, struct_emb, vis_emb):
        return self.alpha * vis_emb + (1 - self.alpha) * struct_emb

class GatedFusion(nn.Module):
    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.gate_layer = nn.Linear(emb_dim * 2, emb_dim)
        
    def forward(self, struct_emb, vis_emb):
        cat = torch.cat([vis_emb, struct_emb], dim=1)
        g = torch.sigmoid(self.gate_layer(cat))
        
        fused = g * vis_emb + (1 - g) * struct_emb
        return fused, g

class MultimodalFusionEngine(nn.Module):
    def __init__(self, struct_in_dim: int, vis_in_dim: int = 2048, hidden_dim: int = 512, out_dim: int = 256):
        super().__init__()
        self.structured_encoder = ProjectionNetwork(struct_in_dim, hidden_dim, out_dim)
        self.visual_encoder = ProjectionNetwork(vis_in_dim, hidden_dim, out_dim)
        
        self.concat_fusion = ConcatFusion(out_dim)
        self.weighted_fusion = WeightedFusion(alpha=0.5)
        self.gated_fusion = GatedFusion(out_dim)
        
    def forward(self, struct_x, vis_x):
        """
        Returns all embeddings + gates for loss and analysis.
        """
        s_emb = self.structured_encoder(struct_x)
        v_emb = self.visual_encoder(vis_x)
        
        s_norm = F.normalize(s_emb, p=2, dim=1)
        v_norm = F.normalize(v_emb, p=2, dim=1)
        
        c_fused = F.normalize(self.concat_fusion(s_norm, v_norm), p=2, dim=1)
        w_fused = F.normalize(self.weighted_fusion(s_norm, v_norm), p=2, dim=1)
        g_fused_raw, gate = self.gated_fusion(s_norm, v_norm)
        g_fused = F.normalize(g_fused_raw, p=2, dim=1)
        
        return {
            "structured": s_norm,
            "visual": v_norm,
            "concat": c_fused,
            "weighted": w_fused,
            "gated": g_fused,
            "gate_values": gate
        }
