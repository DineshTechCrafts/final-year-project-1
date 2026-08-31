import json
from pathlib import Path

ROOT = Path(r"E:\Liver wala thing")
RAW = ROOT / "data" / "raw"

mapping_file = RAW / "series_to_patient.json"
output_file = ROOT / "data" / "download_plan.json"

with open(mapping_file, "r", encoding="utf-8") as f:
    data = json.load(f)

patients = {}

for uid, info in data.items():
    patient = info.get("PatientID")
    modality = info.get("Modality")

    if not patient:
        continue

    patients.setdefault(patient, {"CT": [], "SEG": []})

    record = {
        "series_uid": uid,
        "study_uid": info.get("StudyInstanceUID"),
        "image_count": info.get("ImageCount", 0),
        "description": info.get("SeriesDescription"),
        "path": info.get("path")
    }

    if modality == "CT":
        patients[patient]["CT"].append(record)

    elif modality == "SEG":
        patients[patient]["SEG"].append(record)

plan = []

for patient in sorted(patients):

    segs = patients[patient]["SEG"]
    cts = patients[patient]["CT"]

    if not segs:
        print(f"{patient}: NO SEG")
        continue

    seg = segs[0]

    # The SEG references the source CT series.
    # We cannot infer this relationship from series_to_patient.json alone,
    # so we record all CT candidates for the next verification step.

    plan.append({
        "patient": patient,
        "seg": seg,
        "ct_candidates": sorted(
            cts,
            key=lambda x: x["image_count"],
            reverse=True
        )
    })

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(plan, f, indent=2)

print()
print("DOWNLOAD PLAN CREATED")
print("=====================")
print("Patients:", len(plan))
print("Output:", output_file)
print()

for x in plan[:10]:
    print(
        x["patient"],
        "| SEG:", x["seg"]["series_uid"],
        "| CT candidates:", len(x["ct_candidates"])
    )

print()
print("The plan contains all CT candidates.")
print("Next step will automatically verify SEG → CT references.")