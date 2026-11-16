"""
Tests for validation hyperparameter tuning and ablation utilities.
"""

import copy
import pytest
from src.utils import load_config


def test_config_override_logic():
    base_cfg = load_config("configs/default_config.yaml")
    cfg = copy.deepcopy(base_cfg)

    overrides = {
        "training.learning_rate": 0.0005,
        "models.lstm.hidden_dim": 32,
        "data.lookback_hours": 24
    }

    for key, val in overrides.items():
        parts = key.split(".")
        curr = cfg
        for p in parts[:-1]:
            curr = curr[p]
        curr[parts[-1]] = val

    assert cfg["training"]["learning_rate"] == 0.0005
    assert cfg["models"]["lstm"]["hidden_dim"] == 32
    assert cfg["data"]["lookback_hours"] == 24
    # Ensure base_cfg is not mutated
    assert base_cfg["training"]["learning_rate"] == 0.001


def test_best_candidate_selection_by_mae():
    results = {
        "Exp_1": {"overall": {"MAE": 52.3}},
        "Exp_2": {"overall": {"MAE": 47.1}},
        "Exp_3": {"overall": {"MAE": 49.8}},
    }

    best_exp = min(results.keys(), key=lambda k: results[k]["overall"]["MAE"])
    assert best_exp == "Exp_2"
    assert results[best_exp]["overall"]["MAE"] == 47.1
