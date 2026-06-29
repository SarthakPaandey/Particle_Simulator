"""I/O utilities: load configuration, save data to CSV/JSON."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str) -> None:
    """Save a configuration dictionary as YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def export_bpm_readings(s_positions: np.ndarray, s_bpms: np.ndarray, bpm_values: np.ndarray, path: str) -> None:
    """Export BPM readings as a CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "s_bpm_m": s_bpms,
        "x_mm": bpm_values * 1000.0,
    })
    df.to_csv(path, index=False)


def export_full_trajectory(s: np.ndarray, x: np.ndarray, xp: np.ndarray, path: str) -> None:
    """Export the full beam trajectory (position + angle) as CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "s_m": s,
        "x_mm": x * 1000.0,
        "xp_mrad": xp * 1000.0,
    })
    df.to_csv(path, index=False)


def export_response_matrix(R: np.ndarray, path: str) -> None:
    """Save the response matrix as CSV (n_bpm rows x n_corr cols)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n_bpm, n_corr = R.shape
    df = pd.DataFrame(
        R,
        columns=[f"corrector_{j}" for j in range(n_corr)],
        index=[f"bpm_{i}" for i in range(n_bpm)],
    )
    df.to_csv(path)


def export_response_matrix_npy(R: np.ndarray, path: str) -> None:
    """Save the response matrix as a NumPy .npy file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, R)


def export_simulation_params(params: Dict[str, Any], path: str) -> None:
    """Save simulation parameters as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(params, f, indent=2, default=str)


def load_results_dir(directory: str = "results") -> None:
    """Ensure a results directory exists."""
    os.makedirs(directory, exist_ok=True)


__all__ = [
    "load_config",
    "save_config",
    "export_bpm_readings",
    "export_full_trajectory",
    "export_response_matrix",
    "export_response_matrix_npy",
    "export_simulation_params",
    "load_results_dir",
]
