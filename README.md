# Multi-Agent Deep Learning Framework for Liver Cancer Image Retrieval

An advanced, end-to-end framework utilizing a multi-agent deep learning approach for structurally similar medical image retrieval. Specifically designed for Hepatocellular Carcinoma (HCC), this platform rapidly matches an uploaded CT slice or a queried patient to highly similar reference cases by leveraging semantic segmentation, structured feature extraction, deep visual embeddings, contrastive multimodal fusion, and vector indexing.

> **Research Prototype** — This system is built for research and demonstration purposes. It is not intended for clinical diagnosis or treatment decisions.

---

## 🌟 Key Features

- **End-to-End Real Data Pipeline**: Operates on raw DICOM/NIfTI ingestion and strictly validates against 104 real patient cases (11,277 CT slices).
- **Automated Semantic Segmentation**: Utilizes a robust **U-Net + ResNet-34** backbone to precisely detect and localize Liver, HCC Mass, Portal Vein, and Abdominal Aorta, achieving an impressive **0.7572 Validation Mass Dice**.
- **Deep Multimodal Fusion**:
  - **Structured Agent:** Extracts 39 decorrelated morphological, spatial, and intensity-based features per slice.
  - **Visual Agent:** Extracts deep 2048-dimensional embeddings of the segmented HCC tumor crop using a pretrained **ResNet-50** backbone.
  - **Fusion Engine:** Employs a `GatedFusion` neural network trained via contrastive **InfoNCE loss** (achieving >0.94 MRR), dynamically learning to prioritize visual vs. structural features.
- **Deep Feature Indexing**: Utilizes **FAISS** to rapidly query thousands of fused multi-dimensional representations in milliseconds. Maintains dual indices: a **case-level index** (104 patients) and a **slice-level index** (11,277 slices) for precise 2D-to-2D image matching.
- **Decision Agent**: Local **Llama 3.1 8B** via Ollama synthesizes the retrieved clinical evidence into structured, easy-to-read explanations.
- **Interactive UI**: A beautifully crafted React (Vite) + Tailwind CSS v4 dashboard providing seamless CT analysis and on-the-fly image uploads with end-to-end live inference.

---

## 🏗️ System Architecture

The entire process is heavily decoupled into specialized stages and sub-agents to maximize reproducibility and precision.

```mermaid
flowchart TD
    %% Define Styles
    classDef data fill:#e2e8f0,stroke:#64748b,stroke-width:2px,color:#0f172a;
    classDef agent fill:#bfdbfe,stroke:#3b82f6,stroke-width:2px,color:#0f172a;
    classDef model fill:#fef08a,stroke:#eab308,stroke-width:2px,color:#0f172a;
    classDef db fill:#bbf7d0,stroke:#22c55e,stroke-width:2px,color:#0f172a;

    %% Nodes
    A[(Raw DICOM / NIfTI)]:::data
    B[Preprocessing Agent\nDICOM to NumPy]:::agent
    C[U-Net + ResNet-34\nSemantic Segmentation]:::model
    D{Mask Output\n(Classes 0-4)}:::data
    
    E[Structured Feature Agent\n(Morphology, Intensity, Anatomy)]:::agent
    F[Visual Feature Agent\n(Tumor & Liver Crops)]:::agent
    
    G((39-Dim\nStructured Array)):::data
    H[ResNet-50 Encoder]:::model
    I((2048-Dim\nVisual Embedding)):::data
    
    J[GatedFusion Network\n(Trained via InfoNCE)]:::model
    K((256-Dim\nFused Representation)):::data
    
    L[(FAISS Vector Index\nCase & Slice Level)]:::db
    M[Evidence Reranking Agent]:::agent
    N[Local LLM Decision Agent\n(Llama 3.1 8B)]:::model
    O([Web Dashboard]):::data

    %% Edges
    A --> B
    B --> C
    C --> D
    
    D --> E
    D --> F
    
    E --> G
    F --> H
    H --> I
    
    G --> J
    I --> J
    
    J --> K
    K --> L
    
    L --> M
    M --> N
    N --> O
```

### Segmentation Classes
The segmentation backbone maps CT pixels to the following diagnostic classes:
| ID | Class | Description / Target |
|:---|:---|:---|
| 0 | Background | Slate/None |
| 1 | Liver | Entire Liver Volume |
| 2 | HCC Mass | Hepatocellular Carcinoma Tumor Region |
| 3 | Portal Vein | Proximity tracking for vascular invasion |
| 4 | Abdominal Aorta | Key landmark for anatomical spatial scaling |

---

## 🧠 Pipeline Stages (The "Real" Implementation)

This repository is divided into strict pipeline stages to transform raw medical imaging into a searchable, LLM-interpreted vector space:

1. **Dataset Ingestion & Preprocessing (Stages 1-4):** Converts complex DICOM series into standardized Numpy arrays, strictly isolating patients into Train, Val, and Test splits.
2. **Semantic Segmentation Training:** A U-Net with a ResNet-34 encoder trains on the prepared masks. The model learns to segment the tumor and major organs with high fidelity.
3. **Structured Feature Extraction (Stage 5):** Parses the output masks into exact computational metrics—calculating tumor area, solidity, perimeter, portal vein proximity, and surrounding peritumoral HU intensity gradients (yielding 39 decorrelated features).
4. **Visual Embedding (Stage 6):** Cropping out the segmented HCC tumor and passing it through a ResNet-50 visual encoder to capture deep textural patterns (necrosis, enhancement) that handcrafted features miss.
5. **Contrastive Multimodal Fusion (Stage 7):** A PyTorch `MultimodalFusionEngine` learns to merge the 39-dim structural vector and 2048-dim visual vector. By sampling same-patient slice pairs and training with an `InfoNCELoss`, the model learns an optimal 256-dim fused projection space.
6. **FAISS Retrieval & Multi-Agent Synthesis (Stages 8-10):** The resulting database is queried instantly via FAISS (supporting both case-level queries and real-time slice-level single image uploads). A local LLM agent interprets the retrieval results to explain *why* the matched historical patient is clinically similar to the uploaded scan.

---

## 🛠️ Technology Stack

- **Frontend**: React 18, Vite, Tailwind CSS v4, Recharts, Lucide Icons, TypeScript
- **Backend API**: FastAPI, Uvicorn, Python-Multipart
- **Deep Learning / AI**: PyTorch, Segmentation Models PyTorch (SMP), Torchvision
- **Data & Feature Engineering**: Pandas, NumPy, OpenCV, Parquet
- **Vector Database**: FAISS
- **Local LLM Integration**: Ollama (Llama 3.1 8B)

---

## 📊 Dataset Specifications

Based on the integrated **MedOtter/HCC-TACE-Seg** dataset processing:
- **Total Processed Patients**: 104 
- **Total Valid Slices**: 11,277
- **Train Split**: 72 patients
- **Validation Split**: 15 patients
- **Test Split**: 17 patients

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ (20+ recommended)
- Git
- CUDA-compatible GPU (highly recommended for PyTorch and FAISS)

### 1. Clone the Repository
```bash
git clone https://github.com/DineshTechCrafts/final-year-project-1.git
cd final-year-project-1
```

### 2. Backend Setup
Set up your virtual environment and install the required dependencies:
```bash
# Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install backend dependencies
pip install fastapi uvicorn python-multipart pydantic numpy pillow pandas torch torchvision opencv-python segmentation-models-pytorch faiss-cpu
```

### 3. Frontend Setup
Install the required node modules for the Vite React application:
```bash
cd web
npm install
```

### 4. Running the Application
The platform requires two separate terminal windows.

**Terminal 1: FastAPI Backend**
```bash
.\.venv\Scripts\Activate.ps1
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```
*The backend API will run at `http://localhost:8000`*

**Terminal 2: React Frontend**
```bash
cd web
npm run dev
```
*The web interface will be accessible at `http://localhost:5173`*

---

## 📝 License
This project is released under the **MIT License**. Dataset usage is subject to the original MedOtter HCC-TACE-Seg license agreements. Do not use this for clinical deployment without extensive FDA/CE validation.
