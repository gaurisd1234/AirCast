"""
Utility functions for reproducibility, device selection, and configuration loading.
"""

import json
import os
import random
from typing import Any, Dict
import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    """
    Set seeds across all random number generators to ensure identical results.
    
    Why: Neural network weights and batch shuffling rely on pseudo-random generators.
    Fixing the seed guarantees that every split, initialization, and run is reproducible.
    """
    random.seed(seed)
    # Ensure os-level hash randomization is deterministic if needed
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # If CUDA is available, ensure deterministic kernel operations
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Avoid non-deterministic convolution algorithm choices
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Return torch.device('cuda') if GPU is accessible, otherwise torch.device('cpu').
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML or JSON configuration file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith((".yaml", ".yml")):
            config = yaml.safe_load(f)
        else:
            config = json.load(f)
    return config


def save_json(data: Dict[str, Any], filepath: str, indent: int = 4) -> None:
    """
    Save dictionary as a nicely formatted JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)
