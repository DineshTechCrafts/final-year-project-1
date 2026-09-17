import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\DINESH\Desktop\Liver wala thing\LiverCancer-MultiAgent-Retrieval")
features_dir = PROJECT_ROOT / "data" / "features"

# Load the real output files
slice_gt = pd.read_parquet(features_dir / "slice_features_ground_truth.parquet")
slice_pred = pd.read_parquet(features_dir / "slice_features_predicted.parquet")
case_gt = pd.read_parquet(features_dir / "case_features_ground_truth.parquet")

print("=== BASIC SHAPE CHECKS ===")
print("Slice GT rows:", len(slice_gt), "| unique cases:", slice_gt["case_id"].nunique())
print("Slice Pred rows:", len(slice_pred), "| unique cases:", slice_pred["case_id"].nunique())
print("Case GT rows:", len(case_gt))

print("\n=== IS THIS FAKE DATA? (red flags) ===")
print("Unique case_id values (should be 104, NOT 2 or 3):", slice_gt["case_id"].nunique())
print("Unique tumor_area_px values (should be hundreds+, NOT ~3):", slice_gt["tumor_area_px"].nunique())
print("Any duplicated rows (case_id, slice_index)?:", slice_gt.duplicated(subset=["case_id","slice_index"]).sum())

print("\n=== SANITY: tumor stats only where tumor present ===")
tumor_rows = slice_gt[slice_gt["tumor_present"] == 1]
print("Tumor-positive slice count (should be ~3030):", len(tumor_rows))
print(tumor_rows["tumor_area_px"].describe())

print("\n=== SPOT CHECK: pick one random real case, print a few rows ===")
sample_case = slice_gt["case_id"].sample(1, random_state=1).values[0]
print("Case:", sample_case)
print(slice_gt[slice_gt["case_id"] == sample_case].head(10))

print("\n=== CROSS-CHECK: does GT vs Predicted differ realistically? ===")
merged = slice_gt.merge(slice_pred, on=["case_id","slice_index"], suffixes=("_gt","_pred"))
diff = (merged["tumor_area_px_gt"] - merged["tumor_area_px_pred"]).abs()
print("Mean abs diff GT vs Pred tumor area:", diff.mean())
print("Rows where GT and Pred are EXACTLY identical (suspicious if this is high):", (diff == 0).sum(), "/", len(merged))