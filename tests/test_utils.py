"""
Tests for utility functions: seed reproducibility, config loading, and json saving.
"""

import os
import tempfile
import numpy as np
import torch
from src.utils import get_device, load_config, save_json, set_seed


def test_seed_reproducibility():
    """Verify that set_seed ensures exact reproducibility across runs."""
    set_seed(42)
    np_rand_1 = np.random.rand(10)
    torch_rand_1 = torch.randn(10)

    set_seed(42)
    np_rand_2 = np.random.rand(10)
    torch_rand_2 = torch.randn(10)

    np.testing.assert_array_equal(np_rand_1, np_rand_2)
    assert torch.equal(torch_rand_1, torch_rand_2)


def test_load_config():
    """Verify that default YAML config loads with expected keys."""
    config = load_config("configs/default_config.yaml")
    assert "seed" in config
    assert "data" in config
    assert "training" in config
    assert "models" in config
    assert config["data"]["lookback_hours"] == 72
    assert config["data"]["max_horizon"] == 24


def test_save_json():
    """Verify JSON saving utility."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "test.json")
        payload = {"model": "LSTM", "val_loss": 0.045}
        save_json(payload, test_path)
        assert os.path.exists(test_path)
        loaded = load_config(test_path)
        assert loaded == payload


def test_device_selection():
    """Verify device selection returns a valid torch device."""
    dev = get_device()
    assert isinstance(dev, torch.device)
    assert dev.type in ("cpu", "cuda")
