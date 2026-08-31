"""
Smoke tests for project initialization (Stage 0).
"""
import os
import sys
from pathlib import Path
import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_config
from utils.logger import setup_logger

def test_directories_exist():
    """Verify that essential project directories exist."""
    required_dirs = [
        "data/raw",
        "data/interim",
        "data/processed",
        "configs",
        "models",
        "preprocessing",
        "segmentation",
        "features",
        "agents",
        "retrieval",
        "evaluation",
        "api",
        "frontend",
        "utils",
        "tests"
    ]
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        assert full_path.exists() and full_path.is_dir(), f"Directory not found: {dir_path}"

def test_configuration_loading():
    """Verify that configuration files can be loaded."""
    dataset_cfg_path = PROJECT_ROOT / "configs" / "dataset.yaml"
    
    assert dataset_cfg_path.exists(), "dataset.yaml config is missing."
    
    config = load_config(dataset_cfg_path)
    assert "dataset" in config
    assert "name" in config["dataset"]

def test_logging_utility():
    """Verify that the logger initializes correctly."""
    logger = setup_logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == 20 # logging.INFO

def test_imports():
    """Verify that core modules can be imported without errors."""
    try:
        import utils
        import models
        import preprocessing
        import segmentation
        import features
        import agents
        import retrieval
        import evaluation
    except ImportError as e:
        pytest.fail(f"Failed to import project modules: {e}")
