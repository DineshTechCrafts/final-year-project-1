import faiss
import numpy as np
import pandas as pd
from pathlib import Path

def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds an exact-search inner product (Cosine Similarity) index over normalized embeddings.
    """
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    
    # Ensure C-contiguous float32
    vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
    index.add(vectors)
    return index

def save_index_and_metadata(index: faiss.IndexFlatIP, metadata: pd.DataFrame, 
                            index_path: Path, metadata_path: Path):
    """
    Saves the FAISS index to disk and the corresponding Parquet metadata.
    """
    faiss.write_index(index, str(index_path))
    metadata.to_parquet(metadata_path)
    
def load_index_and_metadata(index_path: Path, metadata_path: Path):
    index = faiss.read_index(str(index_path))
    metadata = pd.read_parquet(metadata_path)
    return index, metadata
