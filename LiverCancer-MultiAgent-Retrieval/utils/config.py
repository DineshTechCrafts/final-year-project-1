"""
Configuration loading utilities.
"""
import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file.
        
    Returns:
        A dictionary containing the configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config
