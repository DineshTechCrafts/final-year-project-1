import os
import sys
import json
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import segmentation_models_pytorch as smp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from segmentation.dataset import HCCTACESegDataset
from segmentation.transforms import get_transforms
from segmentation.losses import CrossEntropyDiceLoss
from segmentation.metrics import calculate_metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["A", "B", "C"], required=True, 
                        help="A: Smoke Test, B: Validation Run, C: Full Baseline")
    args = parser.parse_args()
    
    with open(PROJECT_ROOT / "configs" / "segmentation_baseline.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    train_transform = get_transforms(is_train=True, image_size=config["training"]["image_size"])
    val_transform = get_transforms(is_train=False, image_size=config["training"]["image_size"])
    
    train_dataset = HCCTACESegDataset(
        root_dir=PROJECT_ROOT / config["dataset"]["root_dir"],
        split_file=PROJECT_ROOT / config["dataset"]["split_file"],
        split_name="train",
        transform=train_transform
    )
    
    val_dataset = HCCTACESegDataset(
        root_dir=PROJECT_ROOT / config["dataset"]["root_dir"],
        split_file=PROJECT_ROOT / config["dataset"]["split_file"],
        split_name="validation",
        transform=val_transform
    )
    
    # Subsample for experiments
    if args.experiment == "A":
        print("EXPERIMENT A - Smoke Test")
        # For the smoke test, we just use whatever is in the cache since we haven't bulk downloaded
        all_cached = list((PROJECT_ROOT / config["dataset"]["root_dir"]).glob("*"))
        valid_cases = []
        for case_dir in all_cached:
            meta_path = case_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                num_slices = meta["num_slices"]
                for i in range(num_slices):
                    valid_cases.append({
                        "case_dir": case_dir,
                        "slice_index": i,
                        "patient_id": meta["internal_patient_id"],
                        "case_id": meta["internal_case_id"],
                        "ct_series_uid": meta["ct_series_uid"]
                    })
        train_dataset.samples = valid_cases[:10]
        val_dataset.samples = valid_cases[:10]
        config["training"]["epochs"] = 2
    elif args.experiment == "B":
        print("EXPERIMENT B - Pipeline Validation")
        # Keep only 1 patient
        pid = train_dataset.samples[0]["patient_id"]
        train_dataset.samples = [s for s in train_dataset.samples if s["patient_id"] == pid]
        val_dataset.samples = val_dataset.samples[:10]
        config["training"]["epochs"] = 3
    else:
        print("EXPERIMENT C - Full Baseline Training")
        
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
    
    # Model
    model = smp.Unet(
        encoder_name=config["model"]["encoder"],
        encoder_weights=config["model"]["encoder_weights"],
        in_channels=config["model"]["in_channels"],
        classes=config["model"]["classes"]
    ).to(device)
    
    # Loss & Optimizer
    criterion = CrossEntropyDiceLoss(
        class_weights=config["loss"]["class_weights"],
        ignore_index=config["training"]["ignore_index"]
    )
    optimizer = optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    
    run_dir = PROJECT_ROOT / "runs" / "segmentation_baseline" / f"exp_{args.experiment}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_dice = 0.0
    
    for epoch in range(config["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        
        for images, masks, _ in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, masks, _ in val_loader:
                images, masks = images.to(device), masks.to(device)
                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(masks.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        # Calculate metrics
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        metrics = calculate_metrics(all_preds, all_targets, 
                                    num_classes=config["model"]["classes"], 
                                    ignore_index=config["training"]["ignore_index"])
        
        # Extract mass dice (Class 2)
        mass_dice = metrics["per_class"].get("2", {}).get("dice", 0.0)
        
        print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Mass Dice: {mass_dice:.4f}")
        
        if mass_dice > best_val_dice:
            best_val_dice = mass_dice
            torch.save(model.state_dict(), run_dir / "best_validation_mass_dice.pt")
            with open(run_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=4)
                
    print(f"Training Complete. Best Validation Mass Dice: {best_val_dice:.4f}")

if __name__ == "__main__":
    main()
