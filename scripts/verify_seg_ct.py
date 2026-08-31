import json
from pathlib import Path
import pydicom

ROOT = Path(r"E:\Liver wala thing")
RAW = ROOT / "data" / "raw"
PLAN = ROOT / "data" / "download_plan.json"
OUTPUT = ROOT / "data" / "verified_download_plan.json"

with open(PLAN, "r", encoding="utf-8") as f:
    plan = json.load(f)

verified = []

for i, item in enumerate(plan, 1):

    patient = item["patient"]
    seg = item["seg"]

    print(f"[{i}/{len(plan)}] {patient}")

    # seg["path"] is a directory, so find the DICOM file inside it.
    seg_dir = RAW / seg["path"]
    seg_files = list(seg_dir.glob("*.dcm")) if seg_dir.is_dir() else []

    if not seg_files:
        print("  SEG not downloaded -> cannot verify yet")
        verified.append({
            "patient": patient,
            "status": "SEG_NOT_DOWNLOADED",
            "seg": seg,
            "ct": None
        })
        continue

    seg_path = seg_files[0]

    try:
        ds = pydicom.dcmread(seg_path, stop_before_pixels=True)

        referenced = []

        if hasattr(ds, "ReferencedSeriesSequence"):
            for ref in ds.ReferencedSeriesSequence:
                if hasattr(ref, "SeriesInstanceUID"):
                    referenced.append(str(ref.SeriesInstanceUID))

        if not referenced:
            print("  No referenced CT series found")
            verified.append({
                "patient": patient,
                "status": "NO_REFERENCE",
                "seg": seg,
                "ct": None
            })
            continue

        ref_uid = referenced[0]

        matches = [
            ct for ct in item["ct_candidates"]
            if ct["series_uid"] == ref_uid
        ]

        if matches:
            ct = matches[0]

            print("  VERIFIED")
            print("  CT Series:", ref_uid)
            print("  CT Slices:", ct["image_count"])

            verified.append({
                "patient": patient,
                "status": "VERIFIED",
                "seg": seg,
                "ct": ct
            })

        else:
            print("  CT reference not found in candidates")

            verified.append({
                "patient": patient,
                "status": "REFERENCE_NOT_FOUND",
                "seg": seg,
                "ct": None,
                "referenced_ct_uid": ref_uid
            })

    except Exception as e:
        print("  ERROR:", e)

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(verified, f, indent=2)

print()
print("VERIFICATION COMPLETE")
print("=====================")

counts = {}

for x in verified:
    status = x["status"]
    counts[status] = counts.get(status, 0) + 1

for status, count in counts.items():
    print(status, ":", count)

print()
print("Output:", OUTPUT)