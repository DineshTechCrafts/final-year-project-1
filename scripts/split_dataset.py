import random
import shutil
import json
import re
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

BASE = Path(r"E:\Liver wala thing")

PROCESSED = BASE / "data" / "processed"
SPLIT_ROOT = BASE / "data" / "dataset"

SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6


# ============================================================
# FIND VALID PROCESSED PATIENTS
# ============================================================

patients_by_id = {}

for folder in sorted(PROCESSED.iterdir()):

    if not folder.is_dir():
        continue

    # Accept:
    # hcc001
    # hcc_001
    # HCC001
    # HCC_001

    match = re.fullmatch(
        r"hcc_?(\d+)",
        folder.name.lower()
    )

    if not match:
        continue

    canonical_patient = (
        f"hcc_{int(match.group(1)):03d}"
    )

    image_dir = folder / "images"
    mask_dir = folder / "masks"

    if not image_dir.exists() or not mask_dir.exists():
        continue

    images = sorted(
        image_dir.glob("*.png")
    )

    masks = sorted(
        mask_dir.glob("*.png")
    )

    if not images or not masks:
        continue

    image_names = {
        x.name for x in images
    }

    mask_names = {
        x.name for x in masks
    }

    common = sorted(
        image_names & mask_names
    )

    if not common:
        continue

    # Require equal image/mask counts.
    if len(images) != len(masks):

        print(
            f"WARNING: {folder.name}: "
            f"images={len(images)}, "
            f"masks={len(masks)}"
        )

    patient_info = {
        "patient": canonical_patient,
        "source_folder": folder.name,
        "image_dir": image_dir,
        "mask_dir": mask_dir,
        "files": common
    }

    # --------------------------------------------------------
    # DUPLICATE CANONICAL PATIENT CHECK
    # --------------------------------------------------------

    if canonical_patient in patients_by_id:

        previous = patients_by_id[
            canonical_patient
        ]

        print()
        print("ERROR: DUPLICATE PATIENT DETECTED")
        print(
            f"  Canonical ID: {canonical_patient}"
        )
        print(
            f"  Folder 1: {previous['source_folder']}"
        )
        print(
            f"  Folder 2: {folder.name}"
        )
        print()

        raise RuntimeError(
            f"Duplicate processed patient: "
            f"{canonical_patient}"
        )

    patients_by_id[
        canonical_patient
    ] = patient_info


patients = list(
    patients_by_id.values()
)


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 60)
print("PATIENT-LEVEL DATASET SPLIT")
print("=" * 60)
print()

print(
    "Processed patients available:",
    len(patients)
)

if len(patients) != 104:

    print()
    print(
        "WARNING: Expected 104 unique processed patients."
    )
    print(
        "Found:",
        len(patients)
    )


# ============================================================
# SHUFFLE PATIENTS
# ============================================================

random.seed(SEED)

random.shuffle(patients)

total_patients = len(patients)

train_count = int(
    total_patients * TRAIN_RATIO
)

val_count = int(
    total_patients * VAL_RATIO
)

test_count = (
    total_patients
    - train_count
    - val_count
)


train_patients = patients[
    :train_count
]

val_patients = patients[
    train_count:
    train_count + val_count
]

test_patients = patients[
    train_count + val_count:
]


# ============================================================
# CHECK PATIENT OVERLAP
# ============================================================

train_ids = {
    p["patient"].lower()
    for p in train_patients
}

val_ids = {
    p["patient"].lower()
    for p in val_patients
}

test_ids = {
    p["patient"].lower()
    for p in test_patients
}

overlap_train_val = (
    train_ids & val_ids
)

overlap_train_test = (
    train_ids & test_ids
)

overlap_val_test = (
    val_ids & test_ids
)

if (
    overlap_train_val
    or overlap_train_test
    or overlap_val_test
):

    print()
    print(
        "ERROR: PATIENT LEAKAGE DETECTED"
    )

    print(
        "Train/Val overlap:",
        overlap_train_val
    )

    print(
        "Train/Test overlap:",
        overlap_train_test
    )

    print(
        "Val/Test overlap:",
        overlap_val_test
    )

    raise RuntimeError(
        "Patient appears in more than one dataset split."
    )


# ============================================================
# PRINT PATIENT SPLIT
# ============================================================

print()
print(
    "TRAIN patients:",
    len(train_patients)
)

print(
    "VAL patients:",
    len(val_patients)
)

print(
    "TEST patients:",
    len(test_patients)
)

print()


# ============================================================
# DELETE OLD SPLIT
# ============================================================

print(
    "Removing previous dataset split..."
)

for split in [
    "train",
    "val",
    "test"
]:

    split_dir = (
        SPLIT_ROOT / split
    )

    if split_dir.exists():
        shutil.rmtree(split_dir)

print(
    "Old split removed."
)


# ============================================================
# CREATE CLEAN DIRECTORIES
# ============================================================

for split in [
    "train",
    "val",
    "test"
]:

    (
        SPLIT_ROOT
        / split
        / "images"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    (
        SPLIT_ROOT
        / split
        / "masks"
    ).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# COPY DATA
# ============================================================

def copy_split(
    split_name,
    split_patients
):

    destination_images = (
        SPLIT_ROOT
        / split_name
        / "images"
    )

    destination_masks = (
        SPLIT_ROOT
        / split_name
        / "masks"
    )

    total = 0

    print()
    print(
        f"Creating {split_name.upper()} split..."
    )

    for patient in split_patients:

        patient_name = patient["patient"]

        image_dir = patient["image_dir"]
        mask_dir = patient["mask_dir"]

        files = patient["files"]

        print(
            f"  {patient_name}: "
            f"{len(files)} slices"
        )

        for filename in files:

            source_image = (
                image_dir / filename
            )

            source_mask = (
                mask_dir / filename
            )

            output_name = (
                f"{patient_name}_{filename}"
            )

            shutil.copy2(
                source_image,
                destination_images
                / output_name
            )

            shutil.copy2(
                source_mask,
                destination_masks
                / output_name
            )

            total += 1

    return total


# ============================================================
# EXECUTE SPLITS
# ============================================================

train_images = copy_split(
    "train",
    train_patients
)

val_images = copy_split(
    "val",
    val_patients
)

test_images = copy_split(
    "test",
    test_patients
)


# ============================================================
# FINAL FILE COUNT VALIDATION
# ============================================================

train_image_files = list(
    (
        SPLIT_ROOT
        / "train"
        / "images"
    ).glob("*.png")
)

train_mask_files = list(
    (
        SPLIT_ROOT
        / "train"
        / "masks"
    ).glob("*.png")
)

val_image_files = list(
    (
        SPLIT_ROOT
        / "val"
        / "images"
    ).glob("*.png")
)

val_mask_files = list(
    (
        SPLIT_ROOT
        / "val"
        / "masks"
    ).glob("*.png")
)

test_image_files = list(
    (
        SPLIT_ROOT
        / "test"
        / "images"
    ).glob("*.png")
)

test_mask_files = list(
    (
        SPLIT_ROOT
        / "test"
        / "masks"
    ).glob("*.png")
)


# Images and masks must match.

assert (
    len(train_image_files)
    == len(train_mask_files)
)

assert (
    len(val_image_files)
    == len(val_mask_files)
)

assert (
    len(test_image_files)
    == len(test_mask_files)
)


# ============================================================
# TOTAL SLICE VALIDATION
# ============================================================

total_split_slices = (
    len(train_image_files)
    + len(val_image_files)
    + len(test_image_files)
)

total_processed_slices = sum(
    len(p["files"])
    for p in patients
)


if (
    total_split_slices
    != total_processed_slices
):

    raise RuntimeError(
        f"Slice count mismatch: "
        f"processed={total_processed_slices}, "
        f"split={total_split_slices}"
    )


# ============================================================
# SAVE SPLIT INFORMATION
# ============================================================

split_info = {

    "seed": SEED,

    "ratios": {
        "train": TRAIN_RATIO,
        "validation": VAL_RATIO,
        "test": TEST_RATIO
    },

    "patients": {

        "train": [
            p["patient"]
            for p in train_patients
        ],

        "validation": [
            p["patient"]
            for p in val_patients
        ],

        "test": [
            p["patient"]
            for p in test_patients
        ]
    },

    "slice_counts": {

        "train": len(
            train_image_files
        ),

        "validation": len(
            val_image_files
        ),

        "test": len(
            test_image_files
        ),

        "total": total_split_slices
    },

    "validation": {

        "patient_overlap": False,

        "slice_count_matches_processed": True
    }
}


SPLIT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

split_file = (
    SPLIT_ROOT
    / "split_info.json"
)

with open(
    split_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        split_info,
        f,
        indent=2
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("DATASET SPLIT COMPLETE")
print("=" * 60)
print()

print(
    "Train patients:",
    len(train_patients)
)

print(
    "Validation patients:",
    len(val_patients)
)

print(
    "Test patients:",
    len(test_patients)
)

print()

print(
    "Train slices:",
    len(train_image_files)
)

print(
    "Validation slices:",
    len(val_image_files)
)

print(
    "Test slices:",
    len(test_image_files)
)

print(
    "Total slices:",
    total_split_slices
)

print()

print("Patient overlap:")

print(
    "  Train ∩ Val :",
    len(overlap_train_val)
)

print(
    "  Train ∩ Test:",
    len(overlap_train_test)
)

print(
    "  Val ∩ Test  :",
    len(overlap_val_test)
)

print()

print(
    "Split file:",
    split_file
)

print()
print("=" * 60)