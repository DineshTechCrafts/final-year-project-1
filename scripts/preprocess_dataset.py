import json
import os
import glob
import traceback
import hashlib

import numpy as np
import pydicom
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

_DEFAULT_BASE = r"C:\Users\DINESH\Desktop\Liver wala thing"
BASE = r"E:\Liver wala thing" if os.path.exists(r"E:\Liver wala thing") else _DEFAULT_BASE

PLAN_FILE = os.path.join(
    BASE,
    r"data\verified_download_plan.json"
)

OUT_ROOT = os.path.join(
    BASE,
    r"data\processed"
)

CACHE_CASES_ROOT = os.path.join(
    BASE,
    r"LiverCancer-MultiAgent-Retrieval\data\cache\cases"
)

IMG_SIZE = (256, 256)

CLASS_NAMES = {
    0: "Background",
    1: "Liver",
    2: "Mass",
    3: "Portal vein",
    4: "Aorta"
}


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUT_ROOT, exist_ok=True)


# ============================================================
# LOAD VERIFIED PLAN
# ============================================================

with open(PLAN_FILE, "r", encoding="utf-8") as f:
    plan = json.load(f)

print()
print("=" * 60)
print("FULL HCC-TACE-SEG PREPROCESSING")
print("=" * 60)
print()

print("Patients in verification plan:", len(plan))
print("Output:", OUT_ROOT)
print()


# ============================================================
# GLOBAL STATISTICS
# ============================================================

successful = 0
failed = 0
skipped = 0

total_images = 0
total_masks = 0

failure_log = []


# ============================================================
# PROCESS EACH PATIENT
# ============================================================

for patient_index, item in enumerate(plan, 1):

    patient = item["patient"]
    status = item["status"]

    print()
    print("-" * 60)
    print(f"[{patient_index}/{len(plan)}] {patient}")
    print("-" * 60)

    # --------------------------------------------------------
    # Skip patients without verified CT
    # --------------------------------------------------------

    if status != "VERIFIED":

        print("SKIPPED")
        print("Reason:", status)

        skipped += 1

        failure_log.append({
            "patient": patient,
            "status": "SKIPPED",
            "reason": status
        })

        continue

    try:

        seg_info = item["seg"]
        ct_info = item["ct"]

        # ----------------------------------------------------
        # Locate SEG
        # ----------------------------------------------------

        seg_path = os.path.join(
            BASE,
            r"data\raw",
            seg_info["path"]
        )

        if not os.path.isfile(seg_path):

            # Some Hugging Face downloads may leave the SEG
            # without a .dcm extension.
            possible = glob.glob(
                os.path.join(
                    BASE,
                    r"data\raw",
                    seg_info["path"],
                    "*"
                )
            )

            dcm_files = [
                f for f in possible
                if os.path.isfile(f)
            ]

            if len(dcm_files) == 1:
                seg_path = dcm_files[0]

        if not os.path.isfile(seg_path):

            raise FileNotFoundError(
                f"SEG file not found: {seg_path}"
            )

        print("SEG:", seg_path)

        # ----------------------------------------------------
        # Locate CT directory
        # ----------------------------------------------------

        ct_dir = os.path.join(
            BASE,
            r"data\raw",
            ct_info["path"]
        )

        if not os.path.isdir(ct_dir):

            raise FileNotFoundError(
                f"CT directory not found: {ct_dir}"
            )

        ct_files = sorted(
            glob.glob(
                os.path.join(
                    ct_dir,
                    "*.dcm"
                )
            )
        )

        if not ct_files:

            raise RuntimeError(
                f"No CT DICOM files found: {ct_dir}"
            )

        print("CT files:", len(ct_files))

        # ----------------------------------------------------
        # Load CT
        # ----------------------------------------------------

        ct = []

        for f in ct_files:

            try:
                ds = pydicom.dcmread(
                    f,
                    stop_before_pixels=False
                )

                if hasattr(ds, "SOPInstanceUID"):
                    ct.append(ds)

            except Exception as e:

                print(
                    "  Warning: could not read:",
                    os.path.basename(f)
                )

        if not ct:

            raise RuntimeError(
                "No readable CT slices."
            )

        # ----------------------------------------------------
        # Sort CT slices physically
        # ----------------------------------------------------

        def slice_position(ds):

            if hasattr(
                ds,
                "ImagePositionPatient"
            ):
                try:
                    return float(
                        ds.ImagePositionPatient[2]
                    )
                except Exception:
                    pass

            return float(
                getattr(
                    ds,
                    "InstanceNumber",
                    0
                )
            )

        ct.sort(key=slice_position)

        print(
            "Readable CT slices:",
            len(ct)
        )

        print(
            "CT dimensions:",
            ct[0].Rows,
            "x",
            ct[0].Columns
        )

        # ----------------------------------------------------
        # Load SEG
        # ----------------------------------------------------

        print("Loading SEG...")

        seg = pydicom.dcmread(
            seg_path
        )

        print(
            "SEG frames:",
            getattr(
                seg,
                "NumberOfFrames",
                "UNKNOWN"
            )
        )

        print(
            "SEG dimensions:",
            seg.Rows,
            "x",
            seg.Columns
        )

        # ----------------------------------------------------
        # Segment definitions
        # ----------------------------------------------------

        print("Segments:")

        segment_map = {}

        if hasattr(seg, "SegmentSequence"):

            for s in seg.SegmentSequence:

                number = int(
                    s.SegmentNumber
                )

                label = str(
                    getattr(
                        s,
                        "SegmentLabel",
                        "Unknown"
                    )
                )

                segment_map[number] = label

                print(
                    f"  {number}: {label}"
                )

        # ----------------------------------------------------
        # Read SEG pixels
        # ----------------------------------------------------

        pixel = seg.pixel_array

        print(
            "SEG pixel shape:",
            pixel.shape
        )

        # ----------------------------------------------------
        # Determine frame -> segment
        # ----------------------------------------------------

        frame_segments = []

        for frame in (
            seg.PerFrameFunctionalGroupsSequence
        ):

            try:

                number = int(
                    frame
                    .SegmentIdentificationSequence[0]
                    .ReferencedSegmentNumber
                )

            except Exception:

                number = -1

            frame_segments.append(number)

        frame_segments = np.array(
            frame_segments
        )

        print(
            "SEG segment numbers:",
            np.unique(
                frame_segments
            )
        )

        # ----------------------------------------------------
        # CT SOP UID -> slice index
        # ----------------------------------------------------

        ct_uid_to_index = {}

        for index, ds in enumerate(ct):

            uid = str(
                ds.SOPInstanceUID
            )

            ct_uid_to_index[uid] = index

        # ----------------------------------------------------
        # Create masks
        # ----------------------------------------------------

        masks = np.zeros(
            (
                len(ct),
                seg.Rows,
                seg.Columns
            ),
            dtype=np.uint8
        )

        assigned_frames = 0

        # ----------------------------------------------------
        # Process every SEG frame
        # ----------------------------------------------------

        for frame_index, frame in enumerate(
            seg.PerFrameFunctionalGroupsSequence
        ):

            try:

                segment_number = int(
                    frame
                    .SegmentIdentificationSequence[0]
                    .ReferencedSegmentNumber
                )

            except Exception:

                continue

            # Map actual segment number to our
            # project class IDs.
            #
            # We assume:
            # 1 = liver
            # 2 = mass
            # 3 = portal vein
            # 4 = aorta

            if segment_number not in (
                1,
                2,
                3,
                4
            ):
                continue

            class_id = segment_number

            # ------------------------------------------------
            # Find referenced CT image
            # ------------------------------------------------

            referenced_uid = None

            try:

                source_seq = (
                    frame
                    .DerivationImageSequence[0]
                    .SourceImageSequence
                )

                if source_seq:

                    referenced_uid = str(
                        source_seq[0]
                        .ReferencedSOPInstanceUID
                    )

            except Exception:
                pass

            if (
                referenced_uid is None
                or
                referenced_uid
                not in ct_uid_to_index
            ):
                continue

            slice_index = (
                ct_uid_to_index[
                    referenced_uid
                ]
            )

            frame_data = pixel[
                frame_index
            ]

            masks[
                slice_index
            ][frame_data > 0] = class_id

            assigned_frames += 1

        print(
            "Assigned SEG frames:",
            assigned_frames
        )

        # ----------------------------------------------------
        # Output directories
        # ----------------------------------------------------

        patient_name = patient.lower()

        out_img = os.path.join(
            OUT_ROOT,
            patient_name,
            "images"
        )

        out_mask = os.path.join(
            OUT_ROOT,
            patient_name,
            "masks"
        )

        os.makedirs(
            out_img,
            exist_ok=True
        )

        os.makedirs(
            out_mask,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Export images and masks
        # ----------------------------------------------------

        saved_images = 0
        saved_masks = 0
        patient_vol_imgs = []
        patient_vol_masks = []

        for i, ds in enumerate(ct):

            # ------------------------------------------------
            # CT pixel data
            # ------------------------------------------------

            image = ds.pixel_array.astype(
                np.float32
            )

            slope = float(
                getattr(
                    ds,
                    "RescaleSlope",
                    1
                )
            )

            intercept = float(
                getattr(
                    ds,
                    "RescaleIntercept",
                    0
                )
            )

            image = (
                image * slope
                + intercept
            )

            # ------------------------------------------------
            # Abdominal CT window
            # ------------------------------------------------

            center = 50
            width = 400

            low = (
                center
                - width / 2
            )

            high = (
                center
                + width / 2
            )

            image = np.clip(
                image,
                low,
                high
            )

            image = (
                (
                    image - low
                )
                /
                (
                    high - low
                )
                * 255
            ).astype(
                np.uint8
            )

            # ------------------------------------------------
            # Resize image
            # ------------------------------------------------

            image = Image.fromarray(
                image
            ).resize(
                IMG_SIZE,
                Image.Resampling.BILINEAR
            )

            # ------------------------------------------------
            # Resize segmentation
            # ------------------------------------------------

            mask = Image.fromarray(
                masks[i],
                mode="L"
            ).resize(
                IMG_SIZE,
                Image.Resampling.NEAREST
            )

            # ------------------------------------------------
            # Save
            # ------------------------------------------------

            image_file = os.path.join(
                out_img,
                f"{i:04d}.png"
            )

            mask_file = os.path.join(
                out_mask,
                f"{i:04d}.png"
            )

            image.save(
                image_file
            )

            mask.save(
                mask_file
            )

            saved_images += 1
            saved_masks += 1
            patient_vol_imgs.append(np.array(image, dtype=np.uint8))
            patient_vol_masks.append(np.array(mask, dtype=np.uint8))

        # ----------------------------------------------------
        # Save volumetric .npy and metadata.json for HCCTACESegDataset
        # ----------------------------------------------------
        try:
            pid = hashlib.sha256(patient.encode()).hexdigest()[:12]
            ct_series_uid = ct_info.get("series_uid", "")
            suid_suffix = ct_series_uid[-6:] if len(ct_series_uid) >= 6 else ct_series_uid
            case_id = f"{pid}_{suid_suffix}"
            case_dir = os.path.join(CACHE_CASES_ROOT, case_id)
            os.makedirs(case_dir, exist_ok=True)

            vol_img_arr = np.stack(patient_vol_imgs, axis=0)
            vol_mask_arr = np.stack(patient_vol_masks, axis=0)

            np.save(os.path.join(case_dir, "image.npy"), vol_img_arr)
            np.save(os.path.join(case_dir, "mask.npy"), vol_mask_arr)

            u_labels = np.unique(vol_mask_arr)
            case_meta = {
                "internal_patient_id": pid,
                "internal_case_id": case_id,
                "ct_series_uid": ct_series_uid,
                "num_slices": len(patient_vol_imgs),
                "image_shape": list(vol_img_arr.shape),
                "mask_shape": list(vol_mask_arr.shape),
                "original_patient_id": patient,
                "labels_present": [int(u) for u in u_labels]
            }
            with open(os.path.join(case_dir, "metadata.json"), "w", encoding="utf-8") as f_meta:
                json.dump(case_meta, f_meta, indent=2)
        except Exception as e_vol:
            print("  Warning: could not write volumetric cache:", e_vol)

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        unique, counts = np.unique(
            masks,
            return_counts=True
        )

        class_counts = {}

        for u, c in zip(
            unique,
            counts
        ):

            class_counts[
                int(u)
            ] = int(c)

        # ----------------------------------------------------
        # Patient summary
        # ----------------------------------------------------

        print()
        print("PATIENT COMPLETE")
        print(
            "Images:",
            saved_images
        )

        print(
            "Masks:",
            saved_masks
        )

        print(
            "Classes:",
            class_counts
        )

        # ----------------------------------------------------
        # Verify generated files
        # ----------------------------------------------------

        actual_images = len(
            glob.glob(
                os.path.join(
                    out_img,
                    "*.png"
                )
            )
        )

        actual_masks = len(
            glob.glob(
                os.path.join(
                    out_mask,
                    "*.png"
                )
            )
        )

        if (
            actual_images !=
            actual_masks
        ):

            raise RuntimeError(
                "Image/mask count mismatch"
            )

        successful += 1

        total_images += actual_images
        total_masks += actual_masks

    except Exception as e:

        failed += 1

        print()
        print(
            "ERROR:",
            str(e)
        )

        failure_log.append({
            "patient": patient,
            "status": "FAILED",
            "error": str(e),
            "traceback": traceback.format_exc()
        })


# ============================================================
# SAVE PROCESSING REPORT
# ============================================================

report = {

    "patients_total": len(plan),

    "patients_successful": successful,

    "patients_failed": failed,

    "patients_skipped": skipped,

    "total_images": total_images,

    "total_masks": total_masks,

    "failures": failure_log

}

report_file = os.path.join(
    OUT_ROOT,
    "preprocessing_report.json"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 60)
print("PREPROCESSING FINISHED")
print("=" * 60)

print(
    "Patients total:",
    len(plan)
)

print(
    "Patients successful:",
    successful
)

print(
    "Patients failed:",
    failed
)

print(
    "Patients skipped:",
    skipped
)

print(
    "Total images:",
    total_images
)

print(
    "Total masks:",
    total_masks
)

print()
print(
    "Report:",
    report_file
)

print()
print("=" * 60)