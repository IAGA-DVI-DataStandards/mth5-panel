from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import holoviews as hv
import numpy as np
from holoviews.operation.datashader import datashade


@dataclass(frozen=True)
class RenderConfig:
    plot_height: int = 220
    plot_width: int = 900
    lod_target_points: int = 6_000
    datashade_threshold: int = 2_000_000
    normalize_amplitude: bool = False
    show_hover: bool = False
    use_datashade: bool = False


class LODRenderer:
    """Adaptive renderer for multi-channel xarray payloads."""

    def __init__(self, color_lookup: Dict[str, str] | None = None):
        self.color_lookup = color_lookup or {}

    @staticmethod
    def _lod_stride(
        x: np.ndarray, y: np.ndarray, target_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if target_points <= 0 or y.size <= target_points:
            return x, y
        stride = max(1, y.size // target_points)
        return x[::stride], y[::stride]

    @staticmethod
    def _lod_envelope(
        x: np.ndarray, y: np.ndarray, target_points: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if target_points <= 0 or y.size <= target_points:
            return x, y

        bins = max(1, target_points // 2)
        x_chunks = np.array_split(x, bins)
        y_chunks = np.array_split(y, bins)
        out_x = []
        out_y = []

        for xc, yc in zip(x_chunks, y_chunks):
            if yc.size == 0:
                continue
            i_min = int(np.argmin(yc))
            i_max = int(np.argmax(yc))
            left, right = sorted([i_min, i_max])
            out_x.extend([xc[left], xc[right]])
            out_y.extend([yc[left], yc[right]])

        if not out_x:
            return x, y
        return np.asarray(out_x), np.asarray(out_y)

    def _make_curve(self, key: str, payload: dict, cfg: RenderConfig) -> hv.Curve:
        y = payload["y_normalized"] if cfg.normalize_amplitude else payload["y"]
        xs, ys = self._lod_envelope(payload["x"], y, cfg.lod_target_points)
        if ys.size > cfg.lod_target_points * 3:
            xs, ys = self._lod_stride(xs, ys, cfg.lod_target_points)

        return hv.Curve((xs, ys), kdims=[payload["dim"]], vdims=[payload["vdim"]]).opts(
            color=self.color_lookup.get(key, "#4477AA"),
            height=cfg.plot_height,
            tools=["hover"] if cfg.show_hover else [],
            title=key,
            ylabel=payload.get("units", ""),
            show_grid=True,
            gridstyle={"grid_line_color": "lightgray", "grid_line_alpha": 0.5},
            xticks=12,
        )

    def build_row_plot(
        self, keys: list[str], payloads: Dict[str, dict], cfg: RenderConfig
    ):
        row_payloads = {k: payloads[k] for k in keys if k in payloads}
        if not row_payloads:
            return hv.Curve(([], []))

        use_datashade = cfg.use_datashade and any(
            p.get("n_points", 0) > cfg.datashade_threshold
            for p in row_payloads.values()
        )

        curves = {k: self._make_curve(k, p, cfg) for k, p in row_payloads.items()}
        if not curves:
            return hv.Curve(([], []))

        overlay = hv.NdOverlay(curves, kdims="channel")
        if use_datashade:
            color_key = {k: self.color_lookup.get(k, "#4477AA") for k in curves}
            plot = datashade(
                overlay,
                aggregator="any",
                height=cfg.plot_height,
                width=cfg.plot_width,
                color_key=color_key,
            )
        else:
            plot = overlay

        return plot.opts(
            frame_width=cfg.plot_width,
            framewise=True,
            axiswise=True,
            responsive=False,
        )

    def build_layout(
        self,
        row_map: Dict[int, list[str]],
        payloads: Dict[str, dict],
        cfg: RenderConfig,
    ):
        row_plots = []
        for row_idx in sorted(row_map.keys()):
            keys = row_map[row_idx]
            if keys:
                row_plots.append(self.build_row_plot(keys, payloads, cfg))

        if not row_plots:
            return hv.Curve(([], []))

        return hv.Layout(row_plots).cols(1).opts(shared_axes=True)
