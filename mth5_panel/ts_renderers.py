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
            if np.issubdtype(yc.dtype, np.floating):
                finite = np.isfinite(yc)
                if not finite.any():
                    continue
                yc_work = yc[finite]
                xc_work = xc[finite]
            else:
                yc_work = yc
                xc_work = xc

            if yc_work.size == 0:
                continue

            i_min = int(np.argmin(yc_work))
            i_max = int(np.argmax(yc_work))
            left, right = sorted([i_min, i_max])
            out_x.extend([xc_work[left], xc_work[right]])
            out_y.extend([yc_work[left], yc_work[right]])

        if not out_x:
            return x, y
        return np.asarray(out_x), np.asarray(out_y)

    def _make_curve(self, key: str, payload: dict, cfg: RenderConfig) -> hv.Curve:
        y = payload["y_normalized"] if cfg.normalize_amplitude else payload["y"]
        x_in = np.asarray(payload["x"])
        y_in = np.asarray(y)

        if np.issubdtype(y_in.dtype, np.floating):
            finite = np.isfinite(y_in)
            x_in = x_in[finite]
            y_in = y_in[finite]

        if y_in.size == 0:
            return hv.Curve(([], []), kdims=[payload["dim"]], vdims=[payload["vdim"]])

        xs, ys = self._lod_envelope(x_in, y_in, cfg.lod_target_points)
        if ys.size > cfg.lod_target_points * 3:
            xs, ys = self._lod_stride(xs, ys, cfg.lod_target_points)

        return hv.Curve((xs, ys), kdims=[payload["dim"]], vdims=[payload["vdim"]]).opts(
            color=self.color_lookup.get(key, "#4477AA"),
            height=cfg.plot_height,
            tools=["hover"] if cfg.show_hover else [],
            title="",
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

        curves = {}
        for k, p in row_payloads.items():
            curve = self._make_curve(k, p, cfg)
            if len(curve) > 0:
                curves[k] = curve
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
            plot = overlay.opts(
                legend_position="top",
                legend_cols=max(1, len(curves)),
                show_legend=True,
            )

        return plot.opts(
            frame_width=cfg.plot_width,
            framewise=True,
            axiswise=True,
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
