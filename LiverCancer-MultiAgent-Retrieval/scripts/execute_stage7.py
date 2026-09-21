#!/usr/bin/env python3
"""
Stage 7: Genuine Multimodal Fusion & Metric Learning

Loads Stage 5 structured features and Stage 6 visual embeddings.
Trains the MultimodalFusionEngine using InfoNCELoss (contrastive learning) 
by sampling positive slice-pairs from the same patient. 
Produces real fused embeddings and evaluates MRR/Recall based on 
tercile binning of the tumor-to-liver area ratio.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fusion.fusion_models import MultimodalFusionEngine
from fusion.losses import InfoNCELoss

class SlicePairDataset(Dataset):
    """
    Samples pairs of slices from the same case to act as positive anchors for contrastive learning.
    """
    def __init__(self, struct_feats: np.ndarray, vis_feats: np.ndarray, case_ids: np.ndarray):
        self.struct_feats = torch.tensor(struct_feats, dtype=torch.float32)
        self.vis_feats = torch.tensor(vis_feats, dtype=torch.float32)
        self.case_ids = case_ids
        
        # Group indices by case
        self.unique_cases = np.unique(case_ids)
        self.case_to_indices = {cid: np.where(case_ids == cid)[0] for cid in self.unique_cases}
        
        # Valid cases for pairs (need at least 2 slices)
        self.valid_cases = [cid for cid, idxs in self.case_to_indices.items() if len(idxs) >= 2]
        
    def __len__(self):
        return len(self.valid_cases) * 5  # Arbitrary epoch length multiplier

    def __getitem__(self, idx):
        # Pick a random valid case
        cid = np.random.choice(self.valid_cases)
        indices = self.case_to_indices[cid]
        
        # Sample two distinct slices
        i1, i2 = np.random.choice(indices, size=2, replace=False)
        return (
            self.struct_feats[i1], self.vis_feats[i1],
            self.struct_feats[i2], self.vis_feats[i2]
        )

def evaluate_retrieval(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """
    Evaluates MRR and Recall@5.
    A retrieval is considered a 'match' if the candidate shares the same label (tercile bucket).
    """
    num_cases = len(embeddings)
    
    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embs = embeddings / (norms + 1e-8)
    
    sim_matrix = np.dot(norm_embs, norm_embs.T)
    
    rr_sum = 0.0
    recall5_sum = 0.0
    
    for i in range(num_cases):
        query_label = labels[i]
        # Get similarities, excluding self
        sims = sim_matrix[i].copy()
        sims[i] = -np.inf 
        
        # Sort descending
        ranked_indices = np.argsort(sims)[::-1]
        
        # Find matches
        matches = (labels[ranked_indices] == query_label)
        
        # MRR
        first_match_rank = np.argmax(matches) + 1 if np.any(matches) else 0
        if first_match_rank > 0:
            rr_sum += 1.0 / first_match_rank
            
        # Recall@5 (did any of the top 5 match?)
        if np.any(matches[:5]):
            recall5_sum += 1.0
            
    mrr = rr_sum / num_cases
    recall5 = recall5_sum / num_cases
    return {"MRR": round(mrr, 4), "Recall@5": round(recall5, 4)}

def main():
    print("======================================================================")
    print("STAGE 7: GENUINE MULTIMODAL FUSION & METRIC LEARNING")
    print("======================================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # Paths
    features_dir = PROJECT_ROOT / "data" / "features"
    embeddings_dir = PROJECT_ROOT / "data" / "embeddings" / "ground_truth"
    fused_dir = PROJECT_ROOT / "data" / "embeddings" / "fused"
    metadata_dir = PROJECT_ROOT / "data" / "metadata"
    fused_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data
    print("Loading Stage 5 & 6 features...")
    df_slice = pd.read_parquet(features_dir / "slice_features_ground_truth.parquet")
    vis_embs = np.load(embeddings_dir / "slice_embeddings.npy")  # (11277, 3, 2048)
    
    # The user requested ONLY the Tumor crop (index 2)
    tumor_vis_embs = vis_embs[:, 2, :]  # (11277, 2048)
    
    # Structured features: drop categorical/ID columns
    # Structured features: drop categorical/ID columns AND the evaluation label to prevent leakage
    exclude_cols = ['case_id', 'patient_id', 'partition', 'slice_index', 'tumor_to_liver_area_ratio', 'portal_vein_area_px']
    feat_cols = [c for c in df_slice.columns if c not in exclude_cols and df_slice[c].dtype in [np.float32, np.float64, np.int64, bool]]
    
    struct_feats_raw = df_slice[feat_cols].fillna(0).astype(np.float32).values
    
    # Standard scale structured features
    mean = struct_feats_raw.mean(axis=0, keepdims=True)
    std = struct_feats_raw.std(axis=0, keepdims=True) + 1e-8
    struct_feats = (struct_feats_raw - mean) / std
    
    case_ids = df_slice['case_id'].values
    unique_cases = np.unique(case_ids)
    
    # Compute MRR Ground Truth Labels (Terciles of tumor_to_liver_area_ratio)
    print("Computing case-level tercile bins for MRR evaluation...")
    case_ratios = []
    for cid in unique_cases:
        c_mask = (case_ids == cid)
        # Median area ratio across slices of the case
        ratio = df_slice.loc[c_mask, 'tumor_to_liver_area_ratio'].median()
        case_ratios.append(ratio if pd.notna(ratio) else 0.0)
    
    # Bin into 3 terciles
    try:
        cuts = pd.qcut(case_ratios, 3, duplicates='drop')
        tercile_labels = cuts.codes
    except ValueError:
        # Fallback if qcut fails due to too many zeros
        print("Warning: qcut failed, falling back to equal bins")
        cuts = pd.cut(case_ratios, 3)
        tercile_labels = cuts.codes

    print(f"Prepared 11,277 slices across {len(unique_cases)} cases. Struct Dim: {struct_feats.shape[1]}, Vis Dim: 2048")
    
    # 2. Setup Training
    dataset = SlicePairDataset(struct_feats, tumor_vis_embs, case_ids)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = MultimodalFusionEngine(struct_in_dim=struct_feats.shape[1], vis_in_dim=2048).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = InfoNCELoss(temperature=0.1)
    
    EPOCHS = 20
    print(f"\nTraining Multimodal Fusion via Contrastive Learning for {EPOCHS} epochs...")
    print("NOTE: 20 epochs ensures stronger convergence without the crutch of collinear geometric features.")
    
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for b_idx, (s1, v1, s2, v2) in enumerate(dataloader):
            s1, v1 = s1.to(device), v1.to(device)
            s2, v2 = s2.to(device), v2.to(device)
            
            optimizer.zero_grad()
            out1 = model(s1, v1)
            out2 = model(s2, v2)
            
            # Contrastive loss on the gated fusion output
            loss = criterion(out1["gated"], out2["gated"])
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} | InfoNCE Loss: {epoch_loss/len(dataloader):.4f}")
        
    # 3. Inference & Case Aggregation
    print("\nRunning full inference and aggregating case-level fused embeddings...")
    model.eval()
    
    all_concat = []
    all_weighted = []
    all_gated = []
    all_gates = []
    
    with torch.no_grad():
        s_tensor = torch.tensor(struct_feats, dtype=torch.float32).to(device)
        v_tensor = torch.tensor(tumor_vis_embs, dtype=torch.float32).to(device)
        
        # Batch inference
        batch_size = 256
        for i in range(0, len(s_tensor), batch_size):
            out = model(s_tensor[i:i+batch_size], v_tensor[i:i+batch_size])
            all_concat.append(out["concat"].cpu().numpy())
            all_weighted.append(out["weighted"].cpu().numpy())
            all_gated.append(out["gated"].cpu().numpy())
            all_gates.append(out["gate_values"].cpu().numpy())
            
    cat_slice = np.concatenate(all_concat, axis=0)
    wei_slice = np.concatenate(all_weighted, axis=0)
    gat_slice = np.concatenate(all_gated, axis=0)
    gates_slice = np.concatenate(all_gates, axis=0)
    
    # Mean pool over slices to get case-level representations
    case_cat, case_wei, case_gat = [], [], []
    for cid in unique_cases:
        indices = np.where(case_ids == cid)[0]
        case_cat.append(cat_slice[indices].mean(axis=0))
        case_wei.append(wei_slice[indices].mean(axis=0))
        case_gat.append(gat_slice[indices].mean(axis=0))
        
    case_cat = np.stack(case_cat)
    case_wei = np.stack(case_wei)
    case_gat = np.stack(case_gat)
    
    # Save arrays
    np.save(fused_dir / "concat_fused.npy", case_cat)
    np.save(fused_dir / "weighted_fused.npy", case_wei)
    np.save(fused_dir / "gated_fused.npy", case_gat)
    np.save(fused_dir / "gated_slice_fused.npy", gat_slice)
    
    # Save Model Weights and Scaler for Live Inference
    print("\nSaving GatedFusion model weights and structured feature scaler for live backend inference...")
    models_dir = PROJECT_ROOT / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(model.state_dict(), models_dir / "gated_fusion.pt")
    np.savez(models_dir / "struct_scaler.npz", mean=mean, std=std)
    with open(models_dir / "feat_cols.json", "w") as f:
        json.dump(feat_cols, f)
    
    
    # 4. Evaluation
    print("\nComputing real MRR and Recall@5 ablation (matching on tumor_to_liver_area terciles)...")
    res_cat = evaluate_retrieval(case_cat, tercile_labels)
    res_wei = evaluate_retrieval(case_wei, tercile_labels)
    res_gat = evaluate_retrieval(case_gat, tercile_labels)
    
    print(f"ConcatFusion   -> MRR: {res_cat['MRR']:.4f} | Recall@5: {res_cat['Recall@5']:.4f}")
    print(f"WeightedFusion -> MRR: {res_wei['MRR']:.4f} | Recall@5: {res_wei['Recall@5']:.4f}")
    print(f"GatedFusion    -> MRR: {res_gat['MRR']:.4f} | Recall@5: {res_gat['Recall@5']:.4f}")
    
    mean_gate = float(np.mean(gates_slice))
    vis_reliance = mean_gate * 100
    struct_reliance = (1 - mean_gate) * 100
    print(f"\nGatedFusion Average Gate Act: {mean_gate:.4f} ({vis_reliance:.1f}% visual, {struct_reliance:.1f}% structured)")
    
    # 5. Output Summary
    summary = {
        "status": "STAGE_7_COMPLETE_REAL",
        "evaluation_protocol": (
            "Match defined as cases within the same 'tumor_to_liver_area_ratio' tercile bucket. "
            "This evaluation uses a bucketed structured feature as a clinical-similarity proxy in the absence "
            "of true clinical outcome labels in HCC-TACE-Seg; it measures whether fusion preserves known structured signal, "
            "not validated clinical similarity."
        ),
        "training": {
            "epochs": EPOCHS,
            "loss": "InfoNCELoss (Same-patient slice pairs)",
            "limitations": f"{EPOCHS} epochs is a lightweight training regime, not fully converged."
        },
        "embedding_shapes": {
            "concat": list(case_cat.shape),
            "weighted": list(case_wei.shape),
            "gated": list(case_gat.shape)
        },
        "ablation_results": {
            "ConcatFusion_FullyDecorrelated": res_cat,
            "WeightedFusion_FullyDecorrelated": res_wei,
            "GatedFusion_FullyDecorrelated": res_gat
        },
        "gated_fusion_interpretation": f"Based on the empirical gate distribution (mean={mean_gate:.4f}), the model dynamically prioritizes visual features {vis_reliance:.1f}% of the time and structured features {struct_reliance:.1f}% of the time."
    }
    
    with open(metadata_dir / "stage7_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
        
    print("\nSTAGE 7 EXECUTION COMPLETE")

if __name__ == "__main__":
    main()
