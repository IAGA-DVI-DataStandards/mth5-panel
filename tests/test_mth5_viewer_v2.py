from __future__ import annotations

import numpy as np
import xarray as xr

from mth5_panel.mth5_viewer_v2 import MTH5ViewerV2
from mth5_panel.ts_renderers import LODRenderer, RenderConfig
from mth5_panel.ts_transforms import (
    TransformConfig,
    build_plot_payloads,
    dataset_to_channel_arrays,
)


def test_dataset_to_channel_arrays_and_payloads():
    time = np.arange(0, 10)
    ds = xr.Dataset(
        {
            "ex": xr.DataArray(
                np.linspace(0, 9, 10), dims=["time"], coords={"time": time}
            ),
            "hx": xr.DataArray(
                np.linspace(9, 0, 10), dims=["time"], coords={"time": time}
            ),
        }
    )

    channels = dataset_to_channel_arrays({"s.st.r": ds})
    assert "s.st.r.ex" in channels
    assert "s.st.r.hx" in channels

    payloads = build_plot_payloads(
        channels, TransformConfig(subtract_mean=True, normalize_amplitude=True)
    )
    assert payloads["s.st.r.ex"]["n_points"] == 10
    assert np.all(payloads["s.st.r.ex"]["y_normalized"] >= 0.0)
    assert np.all(payloads["s.st.r.ex"]["y_normalized"] <= 1.0)


def test_lod_renderer_reduces_dense_curve():
    x = np.arange(0, 100_000)
    y = np.sin(x / 1000.0)

    payloads = {
        "s.st.r.ex": {
            "x": x,
            "y": y,
            "y_normalized": (y - y.min()) / (y.max() - y.min()),
            "dim": "time",
            "vdim": "ex",
            "units": "",
            "n_points": int(y.size),
        }
    }

    renderer = LODRenderer(color_lookup={"s.st.r.ex": "#4477AA"})
    layout = renderer.build_layout(
        {1: ["s.st.r.ex"]},
        payloads,
        RenderConfig(
            plot_height=180,
            plot_width=600,
            lod_target_points=2000,
            datashade_threshold=1_000_000,
            normalize_amplitude=False,
            show_hover=False,
            use_datashade=False,
        ),
    )
    assert layout is not None


def test_viewer_v2_refresh_plot_from_channel_arrays():
    viewer = MTH5ViewerV2(use_template=False)
    time = np.arange(0, 1000)
    ds = xr.Dataset(
        {
            "ex": xr.DataArray(
                np.linspace(0, 9, 1000), dims=["time"], coords={"time": time}
            ),
        }
    )

    viewer.channel_arrays = dataset_to_channel_arrays({"a.b.c": ds})
    viewer._assign_colors()
    viewer._build_default_row_assignments()
    viewer._refresh_plot(reason="test")

    assert viewer.plot_pane.object is not None
    assert viewer.plot_refresh_elapsed.value >= 0.0
