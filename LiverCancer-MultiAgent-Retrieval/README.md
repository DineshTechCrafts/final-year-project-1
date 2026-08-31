# Uncertainty-Aware Multi-Agent Deep Learning Framework for Multimodal Liver Cancer Image Retrieval and Clinical Case Matching

## Research Objective
The long-term goal of this research project is to build a system that takes liver cancer medical imaging as input, analyzes the liver and tumor, extracts visual, morphological, anatomical, enhancement, and clinical representations, retrieves clinically similar liver cancer cases, reranks them, estimates retrieval confidence, and provides evidence-grounded explanations.

## Planned Architecture
- **Data Modalities**: Medical imaging (CT/MRI) and clinical data.
- **Agents**: Specialized multi-agent system handling different aspects of feature extraction and retrieval.
- **Encoders**: 3D vision models and clinical data encoders.
- **Fusion & Retrieval**: Advanced fusion mechanisms to combine multimodal features followed by a robust retrieval mechanism leveraging FAISS.

## Current Implementation Status
**STAGE 0 — COMPLETED**
Project structure and development environment have been initialized. No ML models, datasets, or pipelines have been implemented yet.

## Technology Stack
- **Language**: Python 3.11+
- **Deep Learning**: PyTorch, MONAI
- **Medical Imaging**: SimpleITK, nibabel, pydicom
- **Data Science**: NumPy, SciPy, pandas, scikit-learn, OpenCV
- **Vector Search**: FAISS
- **Backend/Frontend**: FastAPI, Streamlit
- **Visualization**: matplotlib, plotly
- **Testing & Config**: pytest, PyYAML

## Directory Structure
- `data/`: Datasets (raw, interim, processed, crops, features, metadata). Excluded from version control.
- `configs/`: YAML configuration files.
- `preprocessing/`: Scripts and modules for data cleaning and preprocessing.
- `segmentation/`: Liver and tumor segmentation modules.
- `features/`: Feature extraction and processing.
- `agents/`: Multi-agent system implementation.
- `models/`: Encoders, fusion models, retrieval models, and custom losses.
- `retrieval/`: Indexing and case matching logic.
- `evaluation/`: Metrics and evaluation scripts.
- `api/`: FastAPI backend implementation.
- `frontend/`: Streamlit web interface.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.
- `scripts/`: Utility and execution scripts.
- `tests/`: Automated tests (pytest).
- `experiments/`: Experiment tracking.
- `logs/`: Application and training logs.
- `checkpoints/`: Model weights.

## Installation Instructions
1. Clone the repository.
2. Ensure you have Python 3.11 or a compatible version.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration
Configuration is kept separate from implementation. We use YAML files stored in the `configs/` directory (e.g., `dataset.yaml`, `model.yaml`, `training.yaml`). 
Paths and sensitive configuration should be managed via environment variables (see `.env.example`).

## Data Privacy & Provenance Principles
- **No patient identifying information (PII/PHI)** will be stored unnecessarily.
- **Data Separation**: Datasets will not be stored inside Git. We enforce strict separation of code and data.
- **Reproducibility**: Experiments will track parameters and seeds to ensure reproducible outcomes.
- **Data Leakage**: We will enforce strict patient-level separation across train/val/test splits.

## Planned Development Stages
- **Stage 0**: Project Initialization (Current)
- **Stage 1**: Dataset Setup and Preprocessing
- **Stage 2**: Segmentation Models
- **Stage 3**: Multimodal Feature Extraction
- **Stage 4**: Retrieval System Development
- **Stage 5**: Multi-Agent Integration
- **Stage 6**: API & Frontend Development
