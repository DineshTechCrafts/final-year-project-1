import json
from pathlib import Path

ROOT = Path(r"E:\Liver wala thing")
RAW = ROOT / "data" / "raw"
OUTPUT = ROOT / "data" / "dataset_plan.json"

mapping_file = RAW / "series_to_patient.json"

if not mapping_file.exists():
    print("ERROR: series_to_patient.json not found")
    raise SystemExit(1)

with open(mapping_file, "r", encoding="utf-8") as f:
    data = json.load(f)

patients = {}

for series_uid, info in data.items():
    patient = info.get("PatientID")
    modality = info.get("Modality")
    image_count = info.get("ImageCount", 0)
    path = info.get("path")

    if not patient:
        continue

    patients.setdefault(patient, {"CT": [], "SEG": []})

    record = {
        "series_uid": series_uid,
        "study_uid": info.get("StudyInstanceUID"),
        "series_description": info.get("SeriesDescription"),
        "image_count": image_count,
        "path": path
    }

    if modality == "CT":
        patients[patient]["CT"].append(record)

    elif modality == "SEG":
        patients[patient]["SEG"].append(record)

plan = []

for patient in sorted(patients):
    ct_series = patients[patient]["CT"]
    seg_series = patients[patient]["SEG"]

    # Prefer CT series with the largest number of slices.
    ct_series = sorted(
        ct_series,
        key=lambda x: x.get("image_count", 0),
        reverse=True
    )

    plan.append({
        "patient": patient,
        "ct_series": ct_series,
        "seg_series": seg_series
    })

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)

print()
print("DATASET PLAN CREATED")
print("====================")
print("Patients found:", len(plan))
print("Output:", OUTPUT)
print()

for item in plan:
    print(
        item["patient"],
        "| CT series:", len(item["ct_series"]),
        "| SEG series:", len(item["seg_series"])
    )