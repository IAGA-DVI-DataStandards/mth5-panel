from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class TransformConfig:
    subtract_mean: bool = True
    normalize_amplitude: bool = False


def _safe_normalize(y: np.ndarray) -> np.ndarray:
    return y / abs(y).max() if y.size else y


def dataset_to_channel_arrays(data_dict: Dict[str, xr.Dataset]) -> Dict[str, dict]:
    """Flatten run datasets into per-channel arrays cached by channel key."""
    channels: Dict[str, dict] = {}

    for run_key, dataset in data_dict.items():
        if isinstance(dataset, xr.DataArray):
            dataset = dataset.to_dataset(name=dataset.name or "amplitude")

        for name, data_array in dataset.data_vars.items():
            component = getattr(data_array, "component", name)
            ch_key = f"{run_key}.{component}"
            dim = list(data_array.dims)[0]
            channels[ch_key] = {
                "x": np.asarray(data_array[dim].values),
                "y_raw": np.asarray(data_array.values),
                "dim": dim,
                "vdim": data_array.name or "amplitude",
                "units": getattr(data_array, "units", ""),
            }

    return channels


def _safe_subtract_mean(y: np.ndarray) -> np.ndarray:
    """Subtract the mean from y, returning a new array."""
    if y.size == 0:
        return y.copy()
    mean_value = float(y.mean())
    print(mean_value)
    return y - mean_value


def build_plot_payloads(
    channel_arrays: Dict[str, dict], config: TransformConfig
) -> Dict[str, dict]:
    """Build per-channel payloads for plotting from cached raw arrays."""
    payloads: Dict[str, dict] = {}

    for key, item in channel_arrays.items():
        y = item["y_raw"]
        if config.subtract_mean:
            y = _safe_subtract_mean(y)

        payloads[key] = {
            "x": item["x"],
            "y": y,
            "y_normalized": _safe_normalize(y),
            "dim": item["dim"],
            "vdim": item["vdim"],
            "units": item["units"],
            "n_points": int(y.size),
        }

    return payloads
