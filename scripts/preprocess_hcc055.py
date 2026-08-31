import glob
import os
import numpy as np
import pydicom
from PIL import Image

BASE = r"E:\Liver wala thing"

CT_DIR = os.path.join(
    BASE,
    r"data\raw\images\HCC_055\1.3.6.1.4.1.14519.5.2.1.1706.8374.192144784182555883228499287181\1.3.6.1.4.1.14519.5.2.1.1706.8374.225555660367207998201981142029"
)

SEG_FILE = glob.glob(
    os.path.join(
        BASE,
        r"data\raw\segmentations\HCC_055\**\*.dcm"
    ),
    recursive=True
)[0]

OUT_IMG = os.path.join(BASE, r"data\processed\hcc055\images")
OUT_MASK = os.path.join(BASE, r"data\processed\hcc055\masks")

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_MASK, exist_ok=True)

print("Loading CT slices...")

ct_files = glob.glob(os.path.join(CT_DIR, "*.dcm"))

if len(ct_files) != 71:
    raise RuntimeError(f"Expected 71 CT slices, found {len(ct_files)}")

ct = [pydicom.dcmread(f) for f in ct_files]

# Sort using physical slice position when available.
ct.sort(
    key=lambda d: float(
        d.ImagePositionPatient[2]
    ) if hasattr(d, "ImagePositionPatient")
    else float(getattr(d, "InstanceNumber", 0))
)

print("CT slices:", len(ct))
print("CT dimensions:", ct[0].Rows, "x", ct[0].Columns)

print("Loading segmentation...")

seg = pydicom.dcmread(SEG_FILE)

print("SEG frames:", seg.NumberOfFrames)
print("SEG dimensions:", seg.Rows, "x", seg.Columns)

print("Segments:")

for item in seg.SegmentSequence:
    print(
        f"  {item.SegmentNumber}: "
        f"{item.SegmentLabel}"
    )

pixel = seg.pixel_array

print("SEG pixel array shape:", pixel.shape)

# DICOM-SEG frames normally have:
#   frame x rows x columns
#
# Segment number for each frame is stored in
# PerFrameFunctionalGroupsSequence.

frame_segments = []

for frame in seg.PerFrameFunctionalGroupsSequence:
    segment_number = (
        frame.SegmentIdentificationSequence[0]
        .ReferencedSegmentNumber
    )
    frame_segments.append(segment_number)

frame_segments = np.array(frame_segments)

print("Unique segment numbers in frames:",
      np.unique(frame_segments))

# Build a mask for each CT slice.
#
# We initially create:
# 0 = background
# 1 = liver
# 2 = mass
# 3 = portal vein
# 4 = aorta

masks = np.zeros(
    (len(ct), seg.Rows, seg.Columns),
    dtype=np.uint8
)

# The SEG references source CT images through
# ReferencedSOPSequence.
#
# Build a mapping from CT SOPInstanceUID to
# its ordered slice index.

ct_uid_to_index = {}

for i, d in enumerate(ct):
    ct_uid_to_index[d.SOPInstanceUID] = i

assigned_frames = 0

for frame_index, frame in enumerate(
    seg.PerFrameFunctionalGroupsSequence
):

    segment_number = (
        frame.SegmentIdentificationSequence[0]
        .ReferencedSegmentNumber
    )

    # Find referenced CT image.
    try:
        source_seq = (
            frame.DerivationImageSequence[0]
            .SourceImageSequence
        )

        referenced_uid = (
            source_seq[0].ReferencedSOPInstanceUID
        )

    except Exception:
        continue

    if referenced_uid not in ct_uid_to_index:
        continue

    slice_index = ct_uid_to_index[referenced_uid]

    frame_data = pixel[frame_index]

    if segment_number == 1:
        class_id = 1
    elif segment_number == 2:
        class_id = 2
    elif segment_number == 3:
        class_id = 3
    elif segment_number == 4:
        class_id = 4
    else:
        continue

    masks[slice_index][frame_data > 0] = class_id

    assigned_frames += 1

print("Assigned SEG frames:", assigned_frames)

# Export 256x256 PNG images and masks.

for i, d in enumerate(ct):

    image = d.pixel_array.astype(np.float32)

    # Apply DICOM rescale.
    slope = float(getattr(d, "RescaleSlope", 1))
    intercept = float(getattr(d, "RescaleIntercept", 0))

    image = image * slope + intercept

    # Window suitable for abdominal CT visualization.
    center = 50
    width = 400

    low = center - width / 2
    high = center + width / 2

    image = np.clip(image, low, high)

    image = (
        (image - low)
        / (high - low)
        * 255
    ).astype(np.uint8)

    image = Image.fromarray(image).resize(
        (256, 256),
        Image.Resampling.BILINEAR
    )

    mask = Image.fromarray(
        masks[i],
        mode="L"
    ).resize(
        (256, 256),
        Image.Resampling.NEAREST
    )

    image_path = os.path.join(
        OUT_IMG,
        f"{i:04d}.png"
    )

    mask_path = os.path.join(
        OUT_MASK,
        f"{i:04d}.png"
    )

    image.save(image_path)
    mask.save(mask_path)

print()
print("PREPROCESSING COMPLETE")
print("Images:", len(glob.glob(os.path.join(OUT_IMG, "*.png"))))
print("Masks:", len(glob.glob(os.path.join(OUT_MASK, "*.png"))))

# Report mask statistics.

unique, counts = np.unique(masks, return_counts=True)

print()
print("MASK CLASS PIXEL COUNTS:")

for u, c in zip(unique, counts):
    names = {
        0: "Background",
        1: "Liver",
        2: "Mass",
        3: "Portal vein",
        4: "Aorta"
    }

    print(
        f"  {u} = {names.get(int(u), 'Unknown')}: "
        f"{int(c)} pixels"
    )