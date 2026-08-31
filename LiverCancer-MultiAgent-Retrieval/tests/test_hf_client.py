"""
Tests for HFDatasetClient.
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.hf_dataset_client import HFDatasetClient

@pytest.fixture
def client(tmp_path):
    return HFDatasetClient(dataset_name="MedOtter/HCC-TACE-Seg", cache_dir=str(tmp_path))

def test_connectivity_and_splits(client):
    """Test API connectivity and splits discovery (Requires Network)."""
    splits = client.get_splits()
    assert len(splits) > 0
    assert "config" in splits[0]
    assert "split" in splits[0]

def test_row_retrieval_and_caching(client, tmp_path):
    """Test retrieving rows and ensuring cache is populated and used."""
    config = "default"
    split = "preview"
    
    # Initial fetch (network)
    rows_data = client.get_cached_rows(config, split, 0, 1)
    assert "rows" in rows_data
    assert len(rows_data["rows"]) == 1
    
    # Check cache file was created
    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    
    # Second fetch (cache)
    rows_data_cached = client.get_cached_rows(config, split, 0, 1)
    assert rows_data_cached == rows_data
