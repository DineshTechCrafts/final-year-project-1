---
license: cc-by-4.0
task_categories:
- image-segmentation
tags:
- medical
- ct
- liver
- hepatocellular-carcinoma
- tace
- tumor-segmentation
- dicom
- tcia
pretty_name: HCC-TACE-Seg
size_categories:
- n<1K
dataset_info:
  features:
  - name: patient_id
    dtype: string
  - name: study_uid
    dtype: string
  - name: ct_series_uid
    dtype: string
  - name: seg_series_uid
    dtype: string
  - name: num_ct_slices
    dtype: int32
  - name: slice_index
    dtype: int32
  - name: liver_voxels
    dtype: int64
  - name: mass_voxels
    dtype: int64
  - name: portal_vein_voxels
    dtype: int64
  - name: aorta_voxels
    dtype: int64
  - name: image
    dtype: image
  - name: mask
    dtype: image
  - name: overlay
    dtype: image
  splits:
  - name: preview
    num_bytes: 26847001
    num_examples: 104
  download_size: 26843221
  dataset_size: 26847001
configs:
- config_name: default
  data_files:
  - split: preview
    path: data/preview-*
---

# HCC-TACE-Seg

Multimodality annotated hepatocellular carcinoma data set including pre- and
post-TACE multiphasic contrast-enhanced CT with voxel-level segmentations of
liver, tumor mass, portal vein, and abdominal aorta.

## Dataset Details

| Field | Value |
|---|---|
| Modality | CT (multiphasic contrast-enhanced) |
| Body part | Liver (hepatocellular carcinoma + intrahepatic vessels + abdominal aorta) |
| Task | 3D multi-class segmentation (4 foreground classes) |
| Patients | 105 (68 M / 37 F) |
| Studies | ~210 (pre-TACE + post-TACE per patient) |
| CT series | 572 |
| SEG series | 105 (one STAPLE-fused consensus mask per patient on the pre-TACE study) |
| Images | 51,968 DICOM slices |
| Format | DICOM (images) + DICOM SEG (segmentations) |
| License | CC BY 4.0 |

Each patient was scanned 1–12 weeks (avg ~3 weeks) before their first TACE
procedure (pre-TACE) and at follow-up (~14 weeks post-TACE). Only the
pre-TACE study is segmented. Three independent radiology residents
semi-automatically segmented each case in AMIRA; senior body radiologist
K.M.E. (20+ yrs experience) reviewed all annotations. The released mask is
the **STAPLE-fused consensus** of the three rater segmentations.

## Segmentation Labels

The DICOM SEG file for each patient contains four binary segments:
1. **Liver** — liver parenchyma
2. **Mass** — entire tumor (necrotic + viable HCC combined)
3. **Portal vein** — intrahepatic vessels
4. **Abdominal aorta**

## Structure

```
images/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
segmentations/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
series_to_patient.json                              # series-level metadata index
```

`PatientID` ranges from `HCC_001` to `HCC_105`. Segmentation DICOMs are
DICOM SEG objects and reference their source CT series via the
`ReferencedSeriesSequence` (top-level CT SeriesInstanceUID) and
`PerFrameFunctionalGroupsSequence → DerivationImageSequence →
SourceImageSequence` (per-frame source CT SOPInstanceUID), enabling
loss-less alignment to the matched CT slice grid.

## Source

- TCIA collection: https://www.cancerimagingarchive.net/collection/hcc-tace-seg/
- DOI: `10.7937/TCIA.5FNA-0924`
- Released: 2022-08-17 (now fully public, no registration required)

## Citation

```bibtex
@article{moawad2023hcctaceseg,
  author  = {Moawad, Ahmed W. and Fuentes, David and ElBanan, Mohamed G. and
             Shalaby, Ahmed S. and Guccione, Jeffrey and Kamel, Serageldin
             and Jensen, Corey T. and Elsayes, Khaled M.},
  title   = {Multimodality annotated hepatocellular carcinoma data set
             including pre- and post-TACE with imaging segmentation},
  journal = {Scientific Data},
  volume  = {10},
  number  = {33},
  year    = {2023},
  doi     = {10.1038/s41597-023-01928-3}
}

@misc{hcctaceseg2022tcia,
  author    = {Moawad, A. W. and Fuentes, D. and ElBanan, M. G. and
               Shalaby, A. S. and Guccione, J. and Kamel, S. and
               Jensen, C. T. and Elsayes, K. M.},
  title     = {HCC-TACE-Seg [Dataset]},
  year      = {2022},
  publisher = {The Cancer Imaging Archive},
  doi       = {10.7937/TCIA.5FNA-0924}
}
```
