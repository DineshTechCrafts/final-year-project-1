import os
import sys
import json
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import segmentation_models_pytorch as smp

# Ensure demo_assets directory exists
OUTPUT_DIR = Path("demo_assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PROJECT_ROOT = Path("LiverCancer-MultiAgent-Retrieval")
sys.path.insert(0, str(PROJECT_ROOT))

from segmentation.transforms import get_transforms

def plot_training_curve():
    print("[1] Plotting training curve...")
    csv_path = Path("training_log.csv")
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue', linewidth=2)
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', color='orange', linewidth=2)
    plt.plot(df['epoch'], df['val_mass_dice'], label='Val Mass Dice', color='green', linewidth=2, linestyle='--')
    
    plt.title('Training Progression (exp_C)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=11)
    
    out_path = OUTPUT_DIR / 'training_curve.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    Saved: {out_path} (Source: {csv_path})")

def plot_per_class_metrics():
    print("[2] Plotting per-class metrics...")
    metrics_path = PROJECT_ROOT / "runs" / "segmentation_baseline" / "exp_C" / "metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    
    classes = ["Background", "Liver", "HCC Mass", "Portal Vein", "Aorta"]
    dice_scores = [
        metrics["per_class"]["0"]["dice"],
        metrics["per_class"]["1"]["dice"],
        metrics["per_class"]["2"]["dice"],
        metrics["per_class"]["3"]["dice"],
        metrics["per_class"]["4"]["dice"]
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(classes, dice_scores, color=['gray', 'green', 'red', 'blue', 'orange'])
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')
        
    plt.title('Validation Dice Score per Class', fontsize=14, fontweight='bold')
    plt.ylabel('Dice Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    out_path = OUTPUT_DIR / 'per_class_metrics.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"    Saved: {out_path} (Source: {metrics_path})")

def generate_mask_overlays():
    print("[3] Generating mask overlays...")
    # 1. Load config and model
    config_path = PROJECT_ROOT / "configs" / "segmentation_baseline.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    checkpoint_path = PROJECT_ROOT / "runs" / "segmentation_baseline" / "exp_C" / "best_validation_mass_dice.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=None,
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"]
    ).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # 2. Select 5 tumor-positive slices spanning different sizes/difficulties
    features_path = PROJECT_ROOT / "data" / "features" / "slice_features_ground_truth.parquet"
    df = pd.read_parquet(features_path)
    tumor_slices = df[df['tumor_present'] == True].sort_values('tumor_area_px')
    
    # Sample 5 slices spread across the distribution
    indices = np.linspace(0, len(tumor_slices)-1, 5, dtype=int)
    selected = tumor_slices.iloc[indices]
    
    transform = get_transforms(is_train=False, image_size=config["training"]["image_size"])
    
    # Color map for overlay
    # 0: BG (transparent), 1: Liver (Green), 2: Mass (Red), 3: PV (Blue), 4: Aorta (Yellow)
    colors = np.array([
        [0, 0, 0, 0],           # Background
        [0, 255, 0, 100],       # Liver
        [255, 0, 0, 150],       # Tumor
        [0, 0, 255, 100],       # Portal Vein
        [255, 255, 0, 100]      # Aorta
    ], dtype=np.uint8)
    
    def apply_overlay(img, mask):
        # img is shape (H, W), mask is shape (H, W) values 0-4
        rgb_img = np.stack((img,)*3, axis=-1).astype(float)
        
        # Create color overlay
        color_mask = colors[mask]
        alpha = color_mask[..., 3:4] / 255.0
        
        # Blend
        overlaid = rgb_img * (1 - alpha) + color_mask[..., :3] * alpha
        return overlaid.astype(np.uint8)

    for idx, row in selected.iterrows():
        case_id = row['case_id']
        slice_idx = row['slice_index']
        
        case_dir = PROJECT_ROOT / "data" / "cache" / "cases" / case_id
        img_array = np.load(case_dir / "image.npy")
        mask_array = np.load(case_dir / "mask.npy")
        
        orig_img = img_array[slice_idx]
        orig_mask = mask_array[slice_idx]
        
        # Clamp image for display [0, 255] roughly based on typical HU windowing applied previously, or just use min/max scaling
        disp_img = np.clip(orig_img, 0, 255).astype(np.uint8)
        
        # Run inference
        # Adapt Grayscale to 3-channel for ResNet34
        if len(orig_img.shape) == 2:
            orig_img_3c = np.stack([orig_img, orig_img, orig_img], axis=-1)
        else:
            orig_img_3c = orig_img
            
        # Apply same transform used during training
        transformed = transform(image=orig_img_3c, mask=orig_mask)
        t_img = transformed['image'].unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(t_img)
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
        # We need to resize original disp_img and orig_mask to 256x256 to match the pred_mask
        import cv2
        disp_img_resized = cv2.resize(disp_img, (256, 256), interpolation=cv2.INTER_LINEAR)
        gt_mask_resized = cv2.resize(orig_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        gt_overlay = apply_overlay(disp_img_resized, gt_mask_resized)
        pred_overlay = apply_overlay(disp_img_resized, pred_mask)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(disp_img_resized, cmap='gray')
        axes[0].set_title(f"Original CT Slice\nCase: {case_id} | Slice: {slice_idx}")
        axes[0].axis('off')
        
        axes[1].imshow(gt_overlay)
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(pred_overlay)
        axes[2].set_title("Model Prediction")
        axes[2].axis('off')
        
        # Add legend
        patches = [
            mpatches.Patch(color='green', label='Liver'),
            mpatches.Patch(color='red', label='HCC Mass'),
            mpatches.Patch(color='blue', label='Portal Vein'),
            mpatches.Patch(color='yellow', label='Aorta')
        ]
        fig.legend(handles=patches, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
        
        plt.tight_layout()
        out_path = OUTPUT_DIR / f'mask_overlay_{case_id}_slice_{slice_idx}.png'
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {out_path} (Source: {case_id}, Checkpoint: best_validation_mass_dice.pt)")

if __name__ == "__main__":
    print("Generating Demo Visualizations from Real Data...")
    plot_training_curve()
    plot_per_class_metrics()
    generate_mask_overlays()
    print("Done! All assets saved to demo_assets/")
