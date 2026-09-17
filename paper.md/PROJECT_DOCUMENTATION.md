# Multi-Agent Deep Learning Framework for Liver Cancer Image Retrieval

## Complete Project Documentation

> **Disclaimer:** This is a **research prototype** built for demonstration and academic exploration. It is **not** intended for clinical diagnosis, treatment decisions, or deployment in healthcare settings without full regulatory validation (FDA, CE, etc.).

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Purpose and Research Goals](#2-project-purpose-and-research-goals)
3. [High-Level System Architecture](#3-high-level-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Dataset: MedOtter HCC-TACE-Seg](#5-dataset-medotter-hcc-tace-seg)
6. [Complete Directory Structure](#6-complete-directory-structure)
7. [Data Preparation Pipeline](#7-data-preparation-pipeline)
8. [Demo Web Application](#8-demo-web-application)
9. [Research ML Framework](#9-research-ml-framework)
10. [Segmentation Subsystem](#10-segmentation-subsystem)
11. [Feature Extraction](#11-feature-extraction)
12. [Visual Embeddings and Fusion](#12-visual-embeddings-and-fusion)
13. [FAISS Retrieval Engine](#13-faiss-retrieval-engine)
14. [Multi-Agent Reranking System](#14-multi-agent-reranking-system)
15. [Explanation and Decision Agents](#15-explanation-and-decision-agents)
16. [Inference Pipeline](#16-inference-pipeline)
17. [Configuration Reference](#17-configuration-reference)
18. [API Reference](#18-api-reference)
19. [Frontend Application](#19-frontend-application)
20. [Testing Suite](#20-testing-suite)
21. [Research Stage Pipeline](#21-research-stage-pipeline)
22. [How to Run the Project](#22-how-to-run-the-project)
23. [Current Limitations and Known Gaps](#23-current-limitations-and-known-gaps)
24. [Complete File Reference](#24-complete-file-reference)

---

## 1. Executive Summary

This project is an **end-to-end liver cancer (Hepatocellular Carcinoma / HCC) medical image retrieval system**. Given a CT scan slice, the system identifies structurally and visually similar reference cases from a curated database of 104 patients and 11,277 annotated slices.

The project consists of **two integrated but partially independent layers**:

| Layer | Location | Purpose |
|-------|----------|---------|
| **Demo Web Stack** | `backend/` + `web/` + root `scripts/` | Interactive React dashboard with FastAPI backend for browsing patients, viewing segmentation overlays, uploading CT slices, and running demo-mode similarity search |
| **Research ML Framework** | `LiverCancer-MultiAgent-Retrieval/` | Full research pipeline: HuggingFace dataset ingestion, U-Net segmentation training, structured feature extraction, ResNet50 visual embeddings, FAISS vector indexing, multi-agent evidence reranking, deterministic explanations, and local Ollama LLM decision synthesis |

The demo web application uses **ground-truth segmentation masks** from the dataset and a **lightweight feature-based retrieval index** (`backend/demo_index.json`). The research framework implements the full multi-agent architecture with FAISS, but requires trained model checkpoints and Ollama to be fully operational at inference time.

---

## 2. Project Purpose and Research Goals

### Primary Objective

Build a system that:

1. Accepts liver cancer medical imaging (primarily CT slices) as input
2. Segments anatomical structures (liver, HCC mass, portal vein, abdominal aorta)
3. Extracts visual, morphological, anatomical, and intensity-based representations
4. Retrieves clinically similar liver cancer cases from a reference database
5. Reranks candidates using a multi-agent evidence fusion system
6. Provides evidence-grounded, human-readable explanations of why cases are similar

### Key Design Principles

- **Zero patient leakage:** Train, validation, and test splits are strictly at the patient level with 0% overlap
- **Fail-fast architecture:** The research pipeline raises explicit errors instead of silently falling back to fake/mock data when artifacts are missing
- **Evidence faithfulness:** Explanation agents only describe metrics that exist in structured evidence; they do not fabricate clinical history
- **Local-only LLM:** The Decision Agent uses Ollama (Llama 3.1 8B) locally — no cloud API fallback
- **Deterministic reranking:** Agent scores are computed with explicit mathematical formulas, not black-box heuristics

---

## 3. High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER (Clinical Researcher)                       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │  React Frontend   │         │  Research API     │
          │  (web/ :5173)     │         │  (LC-MAR/ :8001)  │
          └─────────┬─────────┘         └─────────┬─────────┘
                    │                             │
          ┌─────────▼─────────┐         ┌─────────▼─────────┐
          │  FastAPI Backend  │         │  Inference        │
          │  (backend/ :8000) │         │  Pipeline         │
          └─────────┬─────────┘         └─────────┬─────────┘
                    │                             │
     ┌──────────────┼──────────────┐    ┌──────────┼──────────────┐
     │              │              │    │          │              │
┌────▼────┐  ┌──────▼──────┐  ┌───▼───┐ │  ┌───────▼───────┐  ┌──▼──┐
│ Dataset │  │ Segmentation│  │Retriev│ │  │ Modality      │  │FAISS│
│ Service │  │ Service     │  │Service│ │  │ Router        │  │Index│
└────┬────┘  └──────┬──────┘  └───┬───┘ │  └───────┬───────┘  └──┬──┘
     │              │              │     │          │              │
     └──────────────┼──────────────┘     │  ┌───────▼───────┐  ┌──▼──┐
                    │                    │  │ Segmentation  │  │Multi│
          ┌─────────▼─────────┐          │  │ Adapter       │  │Agent│
          │  data/processed/  │          │  └───────────────┘  │Rerank│
          │  (PNG images +    │          │                     └─────┘
          │   masks)          │          │  ┌─────────────────────────┐
          └───────────────────┘          │  │ Decision Agent (Ollama) │
                                         │  └─────────────────────────┘
                                         └─────────────────────────────┘
```

### Processing Flow (Research Pipeline)

```
Upload CT Image
      │
      ▼
Modality Router Agent ──(MRI?)──► BLOCKED (not validated)
      │ (CT)
      ▼
Input Adapter ── validates file, computes SHA-256 hash
      │
      ▼
Segmentation Agent ── U-Net (ResNet34 encoder) ──► semantic mask
      │
      ▼
Feature Extraction ── morphology + anatomy + intensity features
      │
      ▼
Visual Embedding ── ResNet50 ──► 2048-dim L2-normalized vector
      │
      ▼
Multimodal Fusion ── Concat / Weighted / Gated fusion ──► 256-dim vector
      │
      ▼
FAISS Retrieval ── IndexFlatIP cosine search ──► top-N candidates
      │
      ▼
Multi-Agent Reranking
  ├── Morphology Agent
  ├── Anatomy Agent
  ├── Enhancement (Intensity) Agent
  └── Segmentation Quality Agent
      │
      ▼
Deterministic Reranker ── weighted score fusion
      │
      ▼
Explanation Agent ── human-readable evidence summary
      │
      ▼
Decision Agent (Ollama Llama 3.1 8B) ── structured clinical report
```

### Segmentation Classes

| Class ID | Name | Color (Overlay) | Description |
|----------|------|-----------------|-------------|
| 0 | Background | Transparent | Non-anatomical pixels |
| 1 | Liver | Red | Hepatic parenchyma |
| 2 | HCC Mass | Yellow | Hepatocellular carcinoma tumor |
| 3 | Portal Vein | Blue | Portal venous structure |
| 4 | Abdominal Aorta | Green | Aortic vessel |

---

## 4. Technology Stack

### Frontend (`web/`)

| Technology | Version | Role |
|------------|---------|------|
| React | 18.3 | UI framework |
| Vite | 5.4 | Build tool and dev server |
| TypeScript | 5.6 | Type-safe JavaScript |
| Tailwind CSS | 4.3 | Utility-first styling |
| React Router | 7.18 | Client-side routing |
| Recharts | 3.10 | Charts (bar, pie) |
| Lucide React | 1.38 | Icon library |

### Demo Backend (`backend/`)

| Technology | Role |
|------------|------|
| FastAPI | REST API framework |
| Uvicorn | ASGI server |
| Pillow (PIL) | Image processing for overlays |
| NumPy | Array operations for mask coloring |
| python-multipart | File upload handling |

### Research Framework (`LiverCancer-MultiAgent-Retrieval/`)

| Technology | Role |
|------------|------|
| PyTorch | Deep learning (segmentation, embeddings, fusion) |
| Torchvision | ResNet50 pretrained weights |
| MONAI | Medical imaging utilities |
| FAISS (faiss-cpu) | Vector similarity search |
| Albumentations | Data augmentation for segmentation |
| scikit-image | Morphology feature extraction (regionprops) |
| pandas / Parquet | Feature and metadata storage |
| pydicom | DICOM file reading |
| HuggingFace Dataset API | Remote dataset access |
| Ollama + Llama 3.1 8B | Local LLM for decision synthesis |
| pytest | Automated testing (16 test files) |

---

## 5. Dataset: MedOtter HCC-TACE-Seg

### Source

The project uses the **MedOtter/HCC-TACE-Seg** dataset — a publicly available collection of hepatocellular carcinoma CT scans with DICOM segmentation annotations, hosted on HuggingFace.

### Processed Statistics

| Metric | Value |
|--------|-------|
| Total patients processed | 104 |
| Total valid CT slices | 11,277 |
| Train patients | 72 (~70%) |
| Validation patients | 15 (~15%) |
| Test patients | 17 (~15%) |
| Skipped patients | 1 (HCC_048 — no verified CT) |
| Patient leakage | 0% (strict patient-level splits) |
| Image resolution | 256 × 256 pixels (after preprocessing) |
| Random seed for splits | 42 |

### Data Storage Layout

```
data/
├── raw/                          # Original DICOM files (gitignored)
├── verified_download_plan.json   # Verified CT/SEG pairs per patient
├── download_plan.json            # Download plan from HuggingFace
├── dataset_plan.json             # Dataset organization plan
├── processed/                    # Preprocessed PNG data (gitignored)
│   ├── hcc_001/
│   │   ├── images/               # 0000.png, 0001.png, ...
│   │   └── masks/                # Matching segmentation masks
│   ├── hcc_002/
│   └── preprocessing_report.json
└── dataset/                      # Train/val/test split copies
    ├── split_info.json           # Patient lists and slice counts
    ├── train/images/             # hcc_XXX_0000.png naming
    ├── train/masks/
    ├── val/images/
    ├── val/masks/
    ├── test/images/
    └── test/masks/
```

### Preprocessing Details

CT DICOM slices are processed with:
- **Hounsfield Unit rescaling** using DICOM RescaleSlope and RescaleIntercept
- **Abdominal CT windowing:** center = 50 HU, width = 400 HU
- **Normalization** to 0–255 uint8 range
- **Bilinear resize** for CT images, **nearest-neighbor resize** for masks (preserves class labels)
- **SEG DICOM parsing:** Per-frame functional groups map segment numbers to CT slices via SOP Instance UID references

---

## 6. Complete Directory Structure

```
Liver wala thing/
│
├── README.md                     # Root project README
├── .gitignore                    # Ignores data/, .venv/, node_modules/, etc.
│
├── backend/                      # FastAPI demo/production API
│   ├── main.py                   # App entry point, CORS, router mount
│   ├── demo_index.json           # Generated retrieval index (gitignored)
│   ├── api/
│   │   └── router.py             # All REST endpoints
│   └── services/
│       ├── dataset_service.py    # Patient indexing from split_info.json
│       ├── image_service.py      # Slice PNG path resolution
│       ├── segmentation_service.py  # Mask serving + colored overlays
│       └── retrieval_service.py  # Demo-mode similarity search
│
├── web/                          # React + Vite frontend
│   ├── index.html
│   ├── package.json
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   ├── eslint.config.js
│   └── src/
│       ├── main.tsx              # React DOM mount
│       ├── App.tsx               # Router + sidebar navigation
│       ├── App.css
│       ├── index.css             # Tailwind v4 globals
│       ├── vite-env.d.ts
│       ├── api/
│       │   └── client.ts         # fetchAPI, image URLs, upload
│       └── pages/
│           ├── Dashboard.tsx     # Stats overview + charts
│           ├── Upload.tsx        # CT upload + inference results
│           ├── Analysis.tsx      # Interactive slice viewer
│           ├── Retrieval.tsx     # Similar case search
│           ├── Patients.tsx      # Searchable patient table
│           ├── Dataset.tsx       # Split distribution + class info
│           ├── Agents.tsx        # Multi-agent architecture diagram
│           └── Status.tsx        # Backend component status
│
├── scripts/                      # Root data pipeline utilities
│   ├── prepare_dataset.py        # Build dataset_plan.json
│   ├── make_download_plan.py     # Create download_plan.json
│   ├── verify_seg_ct.py          # Verify CT/SEG pairs
│   ├── download_verified_dataset.ps1  # PowerShell HF downloader
│   ├── preprocess_dataset.py     # DICOM → PNG full pipeline
│   ├── preprocess_hcc055.py      # One-off HCC_055 preprocessing
│   ├── split_dataset.py          # 70/15/15 patient-level split
│   └── generate_demo_index.py    # Build backend/demo_index.json
│
├── data/                         # (Gitignored) All dataset files
│
├── paper.md/                     # This documentation folder
│   └── PROJECT_DOCUMENTATION.md  # You are reading this file
│
└── LiverCancer-MultiAgent-Retrieval/   # Research ML framework
    ├── README.md
    ├── requirements.txt
    ├── .env.example
    ├── .gitignore
    │
    ├── agents/                   # Multi-agent reranking & explanation
    │   ├── anatomy_agent.py
    │   ├── morphology_agent.py
    │   ├── enhancement_agent.py
    │   ├── segmentation_quality_agent.py
    │   ├── reranking_engine.py
    │   ├── explanation_agent.py
    │   ├── evidence_validator.py
    │   ├── llm_explanation_adapter.py
    │   └── decision_agent.py
    │
    ├── api/                      # Research inference API
    │   ├── main.py
    │   ├── retrieval_service.py
    │   └── routes/
    │       ├── __init__.py
    │       └── inference.py
    │
    ├── configs/                  # YAML configuration files
    │   ├── default.yaml
    │   ├── dataset.yaml
    │   ├── model.yaml
    │   ├── training.yaml
    │   ├── mask_labels.yaml
    │   ├── segmentation_baseline.yaml
    │   └── stage11_frozen_experiment.yaml
    │
    ├── embeddings/
    │   ├── encoder.py            # ResNet50 VisualEncoder
    │   └── crop.py               # Bounding-box crop extraction
    │
    ├── features/
    │   ├── morphology.py
    │   ├── anatomy.py
    │   ├── intensity.py
    │   └── normalization.py
    │
    ├── fusion/
    │   ├── fusion_models.py      # Concat/Weighted/Gated fusion
    │   └── losses.py             # InfoNCE contrastive loss
    │
    ├── inference/
    │   ├── pipeline.py           # End-to-end orchestrator
    │   ├── modality_router.py
    │   ├── input_adapter.py
    │   ├── segmentation_adapter.py
    │   └── __init__.py
    │
    ├── preprocessing/
    │   └── hf_dataset_client.py  # HuggingFace API client
    │
    ├── prompts/
    │   └── decision_prompt.py    # Ollama prompt templates
    │
    ├── retrieval/
    │   ├── index_builder.py      # FAISS index build/save/load
    │   └── retriever.py          # CaseRetriever with post-filtering
    │
    ├── segmentation/
    │   ├── dataset.py            # HCCTACESegDataset PyTorch Dataset
    │   ├── mask_reconstructor.py # RGB → semantic label mapping
    │   ├── losses.py             # CrossEntropy + Dice loss
    │   ├── metrics.py            # Per-class Dice metrics
    │   └── transforms.py         # Albumentations transforms
    │
    ├── utils/
    │   ├── config.py             # YAML loader
    │   └── logger.py
    │
    ├── scripts/                  # Stage execution scripts (30 files)
    ├── tests/                    # pytest suite (16 files)
    ├── reports/                  # Generated retrieval reports
    ├── results/stage11/          # Frozen experiment validation outputs
    ├── artifacts/                # Feature preprocessing manifests
    ├── web-ui/                   # Standalone HTML prototype
    ├── data/                     # Framework-local embeddings, vector_db
    ├── models/                   # Model checkpoint storage (empty until trained)
    ├── evaluation/               # Placeholder
    └── runs/                     # Training run logs
```

---

## 7. Data Preparation Pipeline

The root `scripts/` directory contains the full data ingestion and preprocessing workflow. These scripts must be run in order before the web application can serve data.

### Step 1: `prepare_dataset.py`

Reads raw `series_to_patient.json` mapping and builds `data/dataset_plan.json` organizing which DICOM series belong to which patient.

### Step 2: `make_download_plan.py`

Creates `data/download_plan.json` listing all series that need to be downloaded from HuggingFace per patient.

### Step 3: `verify_seg_ct.py`

Verifies that each patient has matching CT and segmentation DICOM files. Outputs `data/verified_download_plan.json` with status per patient (`VERIFIED`, `SKIPPED`, etc.).

### Step 4: `download_verified_dataset.ps1`

PowerShell script that downloads verified DICOM files from HuggingFace into `data/raw/`.

### Step 5: `preprocess_dataset.py`

The main preprocessing pipeline. For each verified patient:

1. Locates CT DICOM directory and SEG DICOM file
2. Reads and sorts CT slices by `ImagePositionPatient[2]` or `InstanceNumber`
3. Parses SEG `PerFrameFunctionalGroupsSequence` to map segment frames to CT slices
4. Builds per-slice uint8 masks with class IDs 1–4
5. Applies abdominal CT windowing (center=50, width=400 HU)
6. Resizes to 256×256 (bilinear for images, nearest for masks)
7. Saves as `data/processed/{patient}/images/{index:04d}.png` and matching masks
8. Writes `data/processed/preprocessing_report.json` with success/failure stats

### Step 6: `split_dataset.py`

Creates patient-level train/validation/test splits:

- **Seed:** 42 (reproducible)
- **Ratios:** 70% train / 15% validation / 15% test
- **Canonical naming:** All patients normalized to `hcc_XXX` format (3-digit zero-padded)
- **Duplicate detection:** Raises error if two folders map to same canonical ID
- **Leakage check:** Asserts zero overlap between train, val, and test patient sets
- **Output:** Copies files to `data/dataset/{split}/images/` with naming `hcc_XXX_0000.png`
- **Metadata:** Writes `data/dataset/split_info.json`

### Step 7: `generate_demo_index.py`

Builds the demo retrieval index used by the web backend:

- Scans all slices in `data/processed/` (parallelized with multiprocessing)
- For each slice, computes features from image + mask:
  - `mean_intensity`, `std_intensity`
  - `liver_ratio`, `mass_ratio`, `portal_ratio`, `aorta_ratio` (pixel fractions)
- Saves to `backend/demo_index.json` as a JSON array of records

---

## 8. Demo Web Application

### Backend (`backend/`)

#### `main.py`

Creates the FastAPI application:
- Title: "Liver Cancer Multi-Agent Retrieval API"
- CORS enabled for `localhost:5173` and `localhost:3000`
- Mounts API router at `/api` prefix
- Can be run directly: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`

#### `api/router.py` — REST Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Returns `{"status": "ok"}` |
| GET | `/api/stats` | Dataset statistics (patients, splits, classes) |
| GET | `/api/patients` | List all patients with split and slice count |
| GET | `/api/patients/{id}` | Single patient metadata |
| GET | `/api/patients/{id}/slices` | Total slice count for patient |
| GET | `/api/patients/{id}/slice/{idx}/image` | PNG CT slice file |
| GET | `/api/patients/{id}/slice/{idx}/mask` | PNG segmentation mask |
| GET | `/api/patients/{id}/slice/{idx}/overlay` | Colored RGBA overlay (query param `classes`) |
| GET | `/api/retrieval/similar/{id}` | Find similar cases (query param `slice_index`) |
| POST | `/api/upload` | Upload CT image for demo inference |
| GET | `/api/model/status` | Component readiness status |

#### `services/dataset_service.py`

- Loads `data/dataset/split_info.json`
- Scans `data/processed/` for patient directories
- Indexes each patient: ID, split partition, slice count, available classes
- Skips `hcc055` (deleted/invalid patient)
- Provides `get_stats()`, `get_all_patients()`, `get_patient()`

#### `services/image_service.py`

- Resolves slice image path: `data/processed/{patient}/images/{index:04d}.png`
- Raises HTTP 404 if slice not found

#### `services/segmentation_service.py`

- Serves raw mask PNGs from `data/processed/{patient}/masks/`
- `get_colored_mask_png()`: Converts grayscale class mask to RGBA overlay
  - Class 1 (Liver): Red `[255, 50, 50, 100]`
  - Class 2 (Mass): Yellow `[255, 255, 0, 180]`
  - Class 3 (Portal Vein): Blue `[50, 50, 255, 150]`
  - Class 4 (Aorta): Green `[50, 255, 50, 150]`
  - Only renders classes specified in `active_classes` parameter

#### `services/retrieval_service.py` — Demo Mode

Operates in **demo mode** using `backend/demo_index.json`:

**Similarity function:** Weighted Euclidean distance converted to 0–1 similarity:
```
dist = sqrt(1.0 * diff_mean² + 2.0 * diff_liver² + 3.0 * diff_mass² + 1.0 * diff_portal²)
similarity = 1.0 / (1.0 + dist)
```

**`get_similar_cases(query_patient_id, query_slice_index)`:**
1. Finds query features in index
2. Computes similarity against all other slices (excludes same patient)
3. Returns top 3 results from distinct patients

**`process_uploaded_image(image_bytes)`:**
1. Reads real mean intensity from uploaded grayscale image
2. Generates **mock** segmentation ratios (random within typical HCC ranges) since U-Net is not connected
3. Searches index for top 3 similar patients
4. Returns simulated features + results

---

## 9. Research ML Framework

Located in `LiverCancer-MultiAgent-Retrieval/`, this is the full research implementation that goes beyond the demo web stack.

### Key Differences from Demo Stack

| Aspect | Demo (`backend/`) | Research (`LC-MAR/`) |
|--------|-------------------|----------------------|
| Segmentation | Ground-truth masks from dataset | Trained U-Net inference |
| Retrieval | Simple feature similarity | FAISS vector search + multi-agent reranking |
| Embeddings | Not used | ResNet50 2048-dim vectors |
| Explanations | None | Deterministic + Ollama LLM synthesis |
| API | Integrated with React UI | Separate FastAPI at `/api/inference` |

---

## 10. Segmentation Subsystem

### `segmentation/dataset.py` — HCCTACESegDataset

PyTorch Dataset that:
- Loads patient splits from JSON split file
- Iterates cached case directories (`{patient_id}_*`)
- Loads NPY volume arrays (`image.npy`, `mask.npy`) per case
- Returns individual slices with metadata (patient_id, case_id, slice_index)
- Converts grayscale to 3-channel for ResNet compatibility
- Applies Albumentations transforms (nearest interpolation for masks)

### `segmentation/mask_reconstructor.py`

Maps RGB mask colors to semantic class labels using `configs/mask_labels.yaml`:
- Black → 0 (Background)
- Green → 1 (Liver)
- Red → 2 (Mass)
- Yellow → 3 (Portal Vein)
- Blue → 4 (Aorta)

### `segmentation/losses.py` — CrossEntropyDiceLoss

Combined loss: Cross-entropy + Dice loss for multi-class segmentation.

### `segmentation/metrics.py`

Per-class Dice coefficient computation for validation.

### `segmentation/transforms.py`

Albumentations train/validation augmentation pipelines with nearest-neighbor mask interpolation.

### `scripts/train_segmentation.py`

Trains U-Net with ResNet34 encoder backbone on HCC-TACE-Seg data. Configuration in `configs/segmentation_baseline.yaml`.

### `inference/segmentation_adapter.py`

Runtime segmentation adapter:
- Expects checkpoint at `models/segmentation_best.pth`
- **Fails loudly** with `FileNotFoundError("SEGMENTATION_ARTIFACT_UNAVAILABLE")` if missing
- No silent fallback to mock segmentation

---

## 11. Feature Extraction

### Morphology Features (`features/morphology.py`)

For each semantic class in a 2D mask, extracts via `skimage.measure.regionprops`:

| Feature | Description |
|---------|-------------|
| `area_px` | Total pixel area |
| `perimeter` | Boundary perimeter |
| `bbox_width`, `bbox_height` | Bounding box dimensions |
| `bbox_aspect_ratio` | Width / height |
| `equivalent_diameter` | Diameter of circle with same area |
| `centroid_x`, `centroid_y` | Center of mass (NaN if absent) |
| `eccentricity` | Ellipse eccentricity |
| `solidity` | Area / convex hull area |
| `extent` | Area / bounding box area |
| `compactness` | Perimeter² / area |
| `circularity` | 4π × area / perimeter² |
| `component_count` | Number of disconnected regions |

Returns zeros for absent classes; NaN for positional/shape metrics when no pixels exist.

### Anatomical Features (`features/anatomy.py`)

Spatial relationships between structures:

| Feature | Formula |
|---------|---------|
| `tumor_to_liver_area_ratio` | tumor_area / liver_area |
| `tumor_to_liver_centroid_distance` | Normalized Euclidean distance between centroids |
| `tumor_to_portal_centroid_distance` | Tumor to portal vein centroid distance |
| `tumor_to_aorta_centroid_distance` | Tumor to aorta centroid distance |

All distances normalized by image width/height. Returns NaN when either structure is absent.

### Intensity Features (`features/intensity.py`)

Per-class intensity statistics from the CT image:

- `mean`, `median`, `std`, `min`, `max`
- Percentiles: `p10`, `p25`, `p75`, `p90`

**Peritumoral ring analysis** (`extract_peritumoral_intensity`):
- Dilates tumor mask by configurable radius (default 5 pixels)
- Subtracts original tumor mask to get ring region
- Computes `peritumoral_mean`, `peritumoral_median`, `peritumoral_std`

### Feature Normalization (`features/normalization.py`)

`FeatureNormalizer` class:
- Median imputation for missing values
- StandardScaler fit **only on training partition** (prevents data leakage)
- Saved scaler applied to validation and test

---

## 12. Visual Embeddings and Fusion

### Visual Encoder (`embeddings/encoder.py`)

`VisualEncoder` based on **ResNet50** (ImageNet pretrained):
- Removes final FC classification layer
- Output: 2048-dimensional feature vector
- Optional L2 normalization (default: enabled)
- Safe zero-vector handling with `eps=1e-8`

### Crop Extraction (`embeddings/crop.py`)

Extracts bounding-box crops around target mask class for focused visual encoding.

### Fusion Models (`fusion/fusion_models.py`)

**ProjectionNetwork:** Linear → LayerNorm → GELU → Dropout → Linear (projects to 256-dim)

**Three fusion strategies:**

| Model | Method |
|-------|--------|
| `ConcatFusion` | Concatenate structured + visual → MLP projection |
| `WeightedFusion` | `α × visual + (1-α) × structured` (default α=0.5) |
| `GatedFusion` | Learned sigmoid gate: `g × visual + (1-g) × structured` |

**MultimodalFusionEngine:** Combines all three strategies, returns normalized embeddings for each fusion type plus gate values.

### Contrastive Loss (`fusion/losses.py`)

`InfoNCELoss` for self-supervised embedding training (positive/negative pair contrastive learning).

---

## 13. FAISS Retrieval Engine

### Index Builder (`retrieval/index_builder.py`)

- Uses `faiss.IndexFlatIP` (inner product = cosine similarity on L2-normalized vectors)
- Ensures C-contiguous float32 arrays
- Saves index to disk via `faiss.write_index()`
- Saves metadata as Parquet via `metadata.to_parquet()`

### Case Retriever (`retrieval/retriever.py`)

`CaseRetriever` class with integrity invariant: `index.ntotal == len(metadata)`.

**`search_by_embedding(query_vector, top_k, top_n, exclude_query_case_id, exclude_patient_id)`:**
1. FAISS exact search for top-N candidates
2. Post-filters: exclude same case ID and/or same patient ID
3. Returns ranked list with case_id, similarity, patient_id, partition

**`search_by_case_id(case_id, top_k, exclude_same_patient)`:**
1. Looks up embedding via `index.reconstruct()`
2. Delegates to `search_by_embedding()`

This design ensures **zero patient leakage** at retrieval time — a patient's own slices never appear as retrieval results.

---

## 14. Multi-Agent Reranking System

After FAISS returns initial candidates, four specialized agents evaluate each query-candidate pair:

### Morphology Agent (`agents/morphology_agent.py`)

Compares tumor morphological features:
- Primary metric: area similarity using exponential decay: `exp(-|area_q - area_c| / scale)`
- Default scale for area: 500 pixels
- Returns NaN similarity if tumor absent in either case

### Anatomy Agent (`agents/anatomy_agent.py`)

Compares spatial/anatomical positioning:
- Primary metric: `tumor_to_liver_centroid_distance` similarity
- Exponential decay with scale 0.5
- Returns NaN if centroids unavailable

### Enhancement Agent (`agents/enhancement_agent.py`)

Compares intensity/enhancement patterns between tumors (HU distribution similarity).

### Segmentation Quality Agent (`agents/segmentation_quality_agent.py`)

Scores segmentation confidence based on uncertain-pixel percentage. Low quality reduces overall confidence.

### Reranking Engine (`agents/reranking_engine.py`)

**DeterministicReranker** — weighted fusion of all scores:

| Component | Default Weight |
|-----------|---------------|
| FAISS similarity | 0.40 |
| Morphology | 0.25 |
| Anatomy | 0.20 |
| Intensity (Enhancement) | 0.10 |
| Segmentation Quality | 0.05 |

Final score clipped to [0, 1]. NaN agent outputs are skipped (not counted in weighted sum).

**GatedEvidenceFusion** — alternative learned-gate fusion (mock implementation for stage execution).

---

## 15. Explanation and Decision Agents

### Evidence Validator (`agents/evidence_validator.py`)

Validates structured evidence JSON:
- No NaN values in required fields
- All scores bounded in [0, 1]
- Required keys present

### Explanation Agent (`agents/explanation_agent.py`)

**Fully deterministic** — no LLM involved:
1. Sorts agent scores (morphology, anatomy, intensity) by value
2. Identifies strongest and weakest evidence sources
3. Builds human-readable explanation citing specific scores
4. Detects FAISS vs. explicit-feature disagreement
5. Sets confidence to LOW if segmentation quality < 0.6

### LLM Explanation Adapter (`agents/llm_explanation_adapter.py`)

Optional LLM wrapper — **disabled by default**. Raises error if enabled (prevents accidental cloud API usage).

### Decision Agent (`agents/decision_agent.py`)

The most sophisticated agent — uses **local Ollama Llama 3.1 8B**:

**What it does:**
- Reads structured evidence from all upstream agents
- Calls Ollama with JSON-format response constraint
- Synthesizes a structured clinical report

**Required output keys:**
- `summary`, `segmentation_findings`, `feature_findings`
- `retrieval_findings`, `evidence_synthesis`, `similar_cases`
- `limitations`, `clinical_note`

**Safety guards:**
- Filters out similar case IDs not present in evidence (prevents hallucination)
- Language guard blocks forbidden phrases ("definitely has cancer", "start chemotherapy", etc.)
- No cloud API fallback — fails with `OllamaUnavailableError` if local server is down
- Empty retrieval results → explicitly states no similar cases found

**Configuration** (`configs/default.yaml`):
```yaml
decision_agent:
  enabled: true
  provider: ollama
  model: llama3.1:8b
  host: "http://127.0.0.1:11434"
  temperature: 0.1
  max_tokens: 1024
  timeout_seconds: 120
```

---

## 16. Inference Pipeline

### `inference/pipeline.py` — `run_inference(input_path, modality)`

End-to-end orchestration:

1. **ModalityRouter** — CT allowed, MRI blocked (`MRI_PIPELINE_NOT_VALIDATED`)
2. **InputAdapter** — Validates file exists, is readable, computes SHA-256 hash
3. **SegmentationAdapter** — Loads frozen U-Net, runs inference (fails if checkpoint missing)
4. *(Future steps: feature extraction → embedding → FAISS search → reranking → explanation)*

Returns structured response with `query_case_id`, `modality`, `processing_latency_sec`, and `results`.

### `inference/modality_router.py`

Gates modalities:
- `"CT"` → `"CT_PIPELINE_OK"`
- `"MRI"` → `"MRI_PIPELINE_NOT_VALIDATED"` (blocked until rigorously tested)
- Other → `"UNSUPPORTED_MODALITY"`

### `inference/input_adapter.py`

- Checks file existence and readability
- Computes SHA-256 content hash for reproducibility tracking

### Research API (`api/routes/inference.py`)

`POST /api/inference` — accepts uploaded CT image, calls `run_inference()`.

---

## 17. Configuration Reference

### `configs/default.yaml`
Ollama decision agent settings (model, host, temperature, timeout).

### `configs/dataset.yaml`
```yaml
dataset:
  name: "liver_cancer_multimodal"
  modalities: ["CT", "MRI"]
  image_size: [128, 128, 128]
  spacing: [1.0, 1.0, 1.0]
  normalization: "z_score"
```

### `configs/model.yaml`
```yaml
model:
  encoders:
    visual: { architecture: "resnet50_3d", pretrained: true }
    clinical: { architecture: "mlp", hidden_dims: [256, 128] }
  fusion: { type: "attention" }
  retrieval: { metric: "cosine", top_k: 10 }
```

### `configs/training.yaml`
Training hyperparameters: batch size, epochs, learning rate, optimizer settings.

### `configs/mask_labels.yaml`
RGB color → semantic label mapping with distance threshold (40.0):
- Black (0,0,0) → 0 Background
- Green (122,199,120) → 1 Liver
- Red (249,66,66) → 2 Mass
- Yellow (250,200,13) → 3 Portal Vein
- Blue (42,125,209) → 4 Aorta

### `configs/segmentation_baseline.yaml`
U-Net ResNet34 segmentation training configuration.

### `configs/stage11_frozen_experiment.yaml`
Frozen experiment parameters for Stage 11 validation (model hashes, environment capture).

### `.env.example`
```
DATA_ROOT=
API_HOST=0.0.0.0
API_PORT=8000
RANDOM_SEED=42
```

---

## 18. API Reference

### Demo Backend (port 8000)

All endpoints prefixed with `/api`. See Section 8 for full endpoint table.

**Model Status Response:**
```json
{
  "Dataset": {"status": "Ready", "color": "green"},
  "Preprocessing": {"status": "Complete", "color": "green"},
  "Segmentation": {"status": "Available (Ground Truth)", "color": "blue"},
  "Retrieval Engine": {"status": "Demo Mode", "color": "yellow"},
  "Embedding Model": {"status": "Pending Training", "color": "gray"},
  "Multi-Agent Pipeline": {"status": "Prototype", "color": "yellow"}
}
```

**Retrieval Response:**
```json
{
  "query_case_id": "hcc_055",
  "demo_mode": true,
  "results": [
    {
      "candidate_case_id": "hcc_023",
      "similarity": 0.847,
      "matched_slice": 42
    }
  ]
}
```

### Research API (separate instance)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/inference` | Upload CT, run full inference pipeline |

---

## 19. Frontend Application

### Navigation (`App.tsx`)

8-page sidebar navigation with React Router:

| Page | Route | Description |
|------|-------|-------------|
| Dashboard | `/` | Dataset stats, split chart, system status |
| Upload Scan | `/upload` | Drag-and-drop CT upload, run analysis pipeline |
| CT Analysis | `/analysis` | Interactive slice viewer with class toggles |
| Case Retrieval | `/retrieval` | Query similar cases by patient + slice |
| Patients | `/patients` | Searchable patient table with split info |
| Dataset | `/dataset` | Split pie chart, class reference, metadata |
| Multi-Agent System | `/agents` | Static architecture flowchart |
| System Status | `/status` | Backend component readiness table |

### API Client (`api/client.ts`)

```typescript
API_BASE_URL = "http://localhost:8000/api"

fetchAPI(endpoint)           // Generic GET
getImageUrl(patient, slice)  // CT slice PNG URL
getMaskUrl(patient, slice)   // Mask PNG URL
getOverlayUrl(patient, slice, classes)  // Colored overlay URL
uploadImage(file)            // POST multipart upload
```

### Page Details

**Dashboard:** Fetches `/stats` and `/model/status`. Shows 4 stat cards (patients, slices, leakage, classes), split distribution table, Recharts bar chart, and system status grid.

**Upload:** File input (PNG/JPG), preview, "Run Analysis Pipeline" button. Calls `POST /upload`. Displays simulated features (mean intensity, liver ratio, mass ratio) and top 3 similar cases with overlay thumbnails.

**Analysis:** Patient dropdown, slice slider (0 to N-1), view mode toggle (original/mask/overlay), per-class checkbox toggles. Real-time overlay rendering via backend API.

**Retrieval:** Patient + slice index selectors, "Find Similar Cases" button. Shows demo mode badge. Displays query preview and top 3 results with similarity percentages.

**Patients:** Searchable table with patient ID, split badge (color-coded), slice count, class count, and "Analyze" link.

**Dataset:** Summary cards, split distribution with pie chart, semantic class reference grid (IDs 0–4).

**Agents:** Static vertical flowchart showing 6 agents with implementation status badges.

**Status:** Table of backend components with colored status indicators. Shows error state if backend unreachable.

### Styling

- Dark theme: `bg-slate-950` background, `text-slate-50` text
- Accent color: Cyan (`text-cyan-400`, `bg-cyan-600`)
- Tailwind CSS v4 with PostCSS
- Responsive grid layouts (1/2/3/4 column breakpoints)

---

## 20. Testing Suite

Located in `LiverCancer-MultiAgent-Retrieval/tests/` — 16 pytest files:

| Test File | What It Validates |
|-----------|-------------------|
| `test_smoke.py` | Directories exist, configs load, imports work |
| `test_hf_client.py` | HuggingFace dataset client connectivity and caching |
| `test_patient_grouping.py` | Deterministic, reproducible patient splits |
| `test_mask_reconstruction_determinism.py` | RGB→label mapping is deterministic |
| `test_segmentation_patient_leakage.py` | No patient overlap in segmentation splits |
| `test_feature_extraction.py` | Morphology, anatomy, intensity correctness |
| `test_feature_leakage.py` | Scaler fit only on train (no leakage) |
| `test_visual_encoder.py` | ResNet50 output dimensions, zero-input safety |
| `test_fusion_shapes.py` | Fusion model tensor shapes + InfoNCE loss |
| `test_faiss_index_integrity.py` | FAISS index count matches metadata rows |
| `test_retrieval_patient_isolation.py` | Query patient excluded from results |
| `test_reranking.py` | Reranker handles missing evidence, scores in [0,1] |
| `test_explanation_faithfulness.py` | Explanation text matches numerical evidence |
| `test_explanation_no_fabrication.py` | No invented metrics in explanations |
| `test_no_fake_fallback.py` | Pipeline fails loudly, no fake fallback data |

Run tests:
```bash
cd LiverCancer-MultiAgent-Retrieval
pytest tests/ -v
```

---

## 21. Research Stage Pipeline

The research framework is organized into sequential development stages, each with dedicated execution scripts:

| Stage | Script | Purpose |
|-------|--------|---------|
| 2 | `stage2_pipeline.py` | Download and cache cases from HuggingFace |
| 3 | `stage3_pipeline.py` | Semantic label mapping provenance |
| 4 | `execute_stage4.py` / `finalize_stage4.py` | Freeze segmentation model config and artifacts |
| 5 | `execute_stage5.py` | Generate slice-level feature Parquet files |
| 6 | `execute_stage6.py` | Generate visual embeddings and fusion representations |
| 7 | `execute_stage7.py` | Build FAISS vector DB indexes |
| 8 | `execute_stage8.py` | FAISS benchmark results |
| 9 | `execute_stage9.py` | Multi-agent reranking final results |
| 11 | `execute_stage11.py` | Frozen experiment validation (model hashes, environment) |
| 11 | `run_stage11_final_validation.py` | Gate check for all Stage 11 result files |
| 12 | `audit_stage12_dependencies.py` | Dependency audit with file hashes |

**Supporting scripts:**
- `inspect_remote_dataset.py` — Inspect HF dataset, generate patient splits
- `recover_dataset.py` — Re-download cached case data
- `investigate_label_semantics.py` — Mask color→label provenance
- `train_segmentation.py` — Train U-Net segmentation model
- `demo_real_ct.py` — CLI inference demo on real CT image
- `test_end_to_end_inference.py` — MRI blocking + CT success paths
- `test_decision_agent.py` — Standalone decision agent test
- `discover_frozen_models.py` — Hash frozen model checkpoints
- `validate_frozen_runtime.py` — Validate all runtime artifacts present
- `inventory_inference_artifacts.py` — Inventory and hash pipeline artifacts
- `generate_retrieval_report.py` / `generate_html_report.py` — Report generation
- `generate_stage11_html.py` — Stage 11 HTML report from JSON
- `check_stats.py` — Print slice counts per split

**Stage 11 Results** (`results/stage11/`):
- `stage11_benchmark.json` — Retrieval benchmark metrics
- `stage11_ablation.json` — Ablation study results
- `stage11_statistical_analysis.json` — Statistical significance tests
- `stage11_failure_analysis.json` — Failure case analysis
- `stage11_summary.json` — Overall summary
- `model_hashes.json` — SHA hashes of frozen model files
- `environment.json` — Python/package versions for reproducibility
- `stage11_final_report.html` — Human-readable HTML report

---

## 22. How to Run the Project

### Prerequisites

- Python 3.10+
- Node.js 18+ (20+ recommended)
- Git
- (Optional) Ollama with `llama3.1:8b` for Decision Agent

### Step 1: Python Environment

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows
pip install fastapi uvicorn python-multipart pydantic numpy pillow
```

For the full research framework:
```bash
cd LiverCancer-MultiAgent-Retrieval
pip install -r requirements.txt
```

### Step 2: Data Preparation (if starting from scratch)

```bash
python scripts/prepare_dataset.py
python scripts/make_download_plan.py
python scripts/verify_seg_ct.py
# Run download_verified_dataset.ps1
python scripts/preprocess_dataset.py
python scripts/split_dataset.py
python scripts/generate_demo_index.py
```

### Step 3: Start Backend

```bash
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

API available at: `http://localhost:8000`
API docs at: `http://localhost:8000/docs`

### Step 4: Start Frontend

```bash
cd web
npm install
npm run dev
```

UI available at: `http://localhost:5173`

### Step 5 (Optional): Research Inference API

```bash
cd LiverCancer-MultiAgent-Retrieval
uvicorn api.main:app --host 0.0.0.0 --port 8001
```

### Step 6 (Optional): Decision Agent

```bash
ollama pull llama3.1:8b
cd LiverCancer-MultiAgent-Retrieval
python scripts/test_decision_agent.py
```

---

## 23. Current Limitations and Known Gaps

| Component | Status | Notes |
|-----------|--------|-------|
| Demo retrieval | Working | Feature-based similarity, not FAISS |
| Demo upload inference | Partial | Uses mock segmentation ratios, real intensity only |
| Ground-truth segmentation display | Working | Serves pre-computed masks from dataset |
| U-Net inference (runtime) | Not connected | Checkpoint required at `models/segmentation_best.pth` |
| FAISS retrieval (demo UI) | Not connected | Research framework has full implementation |
| ResNet50 embeddings (demo) | Not connected | "Pending Training" status |
| Multi-agent reranking (demo) | Not connected | Full implementation in research framework |
| Ollama Decision Agent | Requires setup | Needs local Ollama + llama3.1:8b model |
| MRI modality | Blocked | Explicitly rejected by modality router |
| HCC-TACE-Seg DICOM files | Gitignored/deleted | Must be re-downloaded via scripts |
| `preprocess_dataset.py` BASE path | Hardcoded | Points to `E:\Liver wala thing` — update for your machine |
| `split_dataset.py` BASE path | Hardcoded | Same as above |
| Research README | Outdated | Says "Stage 0 — COMPLETED" but stages 2–12 exist |

---

## 24. Complete File Reference

### Root Level (3 files)

| File | Purpose |
|------|---------|
| `README.md` | Project overview, installation, architecture diagram |
| `.gitignore` | Excludes data/, .venv/, node_modules/, checkpoints |
| `paper.md/PROJECT_DOCUMENTATION.md` | This comprehensive documentation |

### `backend/` (6 Python files + 1 JSON)

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | 25 | FastAPI app entry, CORS, router mount |
| `api/router.py` | 86 | 11 REST endpoints |
| `services/dataset_service.py` | 75 | Patient indexing from split_info.json |
| `services/image_service.py` | 21 | Slice PNG path resolution |
| `services/segmentation_service.py` | 56 | Mask serving + RGBA overlay generation |
| `services/retrieval_service.py` | 172 | Demo similarity search + upload processing |
| `demo_index.json` | Generated | Per-slice feature index for demo retrieval |

### `web/` (18 source files)

| File | Purpose |
|------|---------|
| `index.html` | Vite HTML shell |
| `package.json` | Dependencies and npm scripts |
| `vite.config.ts` | Vite dev server config |
| `tailwind.config.js` | Tailwind configuration |
| `postcss.config.js` | PostCSS + Tailwind plugin |
| `eslint.config.js` | ESLint flat config |
| `tsconfig.json` / `tsconfig.app.json` / `tsconfig.node.json` | TypeScript configs |
| `src/main.tsx` | React DOM mount point |
| `src/App.tsx` | Router + 8-page sidebar navigation |
| `src/App.css` | App-level styles |
| `src/index.css` | Tailwind v4 global styles |
| `src/vite-env.d.ts` | Vite type declarations |
| `src/api/client.ts` | API client with 5 exported functions |
| `src/pages/Dashboard.tsx` | Stats overview with Recharts bar chart |
| `src/pages/Upload.tsx` | CT upload + inference results display |
| `src/pages/Analysis.tsx` | Interactive slice viewer with class toggles |
| `src/pages/Retrieval.tsx` | Similar case search interface |
| `src/pages/Patients.tsx` | Searchable patient data table |
| `src/pages/Dataset.tsx` | Split pie chart + class reference |
| `src/pages/Agents.tsx` | Static multi-agent architecture diagram |
| `src/pages/Status.tsx` | Backend component status panel |

### `scripts/` (8 files)

| File | Purpose |
|------|---------|
| `prepare_dataset.py` | Build dataset_plan.json |
| `make_download_plan.py` | Create download_plan.json |
| `verify_seg_ct.py` | Verify CT/SEG pairs |
| `download_verified_dataset.ps1` | PowerShell HF downloader |
| `preprocess_dataset.py` | Full DICOM→PNG pipeline (811 lines) |
| `preprocess_hcc055.py` | One-off HCC_055 preprocessing |
| `split_dataset.py` | Patient-level 70/15/15 split (694 lines) |
| `generate_demo_index.py` | Build demo_index.json with multiprocessing |

### `LiverCancer-MultiAgent-Retrieval/agents/` (9 files)

| File | Purpose |
|------|---------|
| `anatomy_agent.py` | Anatomical similarity scoring |
| `morphology_agent.py` | Morphological similarity scoring |
| `enhancement_agent.py` | Intensity/enhancement similarity |
| `segmentation_quality_agent.py` | Segmentation confidence scoring |
| `reranking_engine.py` | DeterministicReranker + GatedEvidenceFusion |
| `explanation_agent.py` | Deterministic explanation generation |
| `evidence_validator.py` | Evidence JSON schema validation |
| `llm_explanation_adapter.py` | Optional LLM wrapper (disabled) |
| `decision_agent.py` | Ollama Llama 3.1 8B evidence synthesis (363 lines) |

### `LiverCancer-MultiAgent-Retrieval/` — All Other Modules

| Directory | Files | Purpose |
|-----------|-------|---------|
| `api/` | 4 | Research FastAPI (main, routes, retrieval_service) |
| `configs/` | 7 YAML | All configuration files |
| `embeddings/` | 2 | ResNet50 encoder + crop extraction |
| `features/` | 4 | Morphology, anatomy, intensity, normalization |
| `fusion/` | 2 | Multimodal fusion models + InfoNCE loss |
| `inference/` | 5 | Pipeline orchestration + adapters |
| `preprocessing/` | 1 | HuggingFace dataset client |
| `prompts/` | 1 | Ollama decision prompt templates |
| `retrieval/` | 2 | FAISS index builder + case retriever |
| `segmentation/` | 6 | Dataset, masks, losses, metrics, transforms |
| `utils/` | 2 | Config loader + logger |
| `scripts/` | 30 | Stage pipelines + utilities |
| `tests/` | 16 | pytest test suite |
| `reports/` | 2 | Generated retrieval reports (JSON + HTML) |
| `results/stage11/` | 8 | Frozen experiment validation outputs |
| `web-ui/` | 1 | Standalone HTML prototype for research API |

---

## Summary

This project implements a **multi-agent deep learning framework for liver cancer CT image retrieval** with two layers:

1. A **production-ready demo web application** (React + FastAPI) for browsing 104 patients, viewing segmentation overlays, uploading CT slices, and running feature-based similarity search.

2. A **comprehensive research framework** with U-Net segmentation, ResNet50 embeddings, FAISS vector search, four specialized reranking agents, deterministic explanations, and local Ollama LLM decision synthesis — designed with strict patient-level data separation, fail-fast error handling, and evidence faithfulness guarantees.

The system processes the **MedOtter HCC-TACE-Seg** dataset (11,277 slices, 5 semantic classes) through a rigorous DICOM preprocessing pipeline and serves results through an interactive dark-themed medical imaging dashboard.

---

*Documentation generated from complete source code analysis. Last updated: September 2026.*
