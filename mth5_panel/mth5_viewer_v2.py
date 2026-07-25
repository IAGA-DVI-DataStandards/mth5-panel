from __future__ import annotations

import pathlib
import sys
import time

import colorcet as cc
import holoviews as hv
import pandas as pd
import panel as pn
import param
from bokeh.palettes import Viridis256

try:
    from .ts_data_store import MTDataStore
    from .ts_renderers import LODRenderer, RenderConfig
    from .ts_transforms import (
        TransformConfig,
        build_plot_payloads,
        dataset_to_channel_arrays,
    )
except ImportError:
    # Support direct execution via: panel serve mth5_viewer_v2.py
    this_dir = pathlib.Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))
    from ts_data_store import MTDataStore
    from ts_renderers import LODRenderer, RenderConfig
    from ts_transforms import (
        TransformConfig,
        build_plot_payloads,
        dataset_to_channel_arrays,
    )

pn.extension("tabulator", sizing_mode="stretch_width")
hv.extension("bokeh")

RUN_SUMMARY_DISPLAY_COLUMNS = [
    "survey",
    "station",
    "run",
    "start",
    "end",
    "n_samples",
    "sample_rate",
    "input_channels",
    "output_channels",
]


def empty_curve():
    return hv.Curve(([], []), kdims=["time"], vdims=["amplitude"]).opts(
        frame_width=1,
        height=1,
        toolbar=None,
    )


class MTH5ViewerV2(param.Parameterized):
    """V2 xarray-first viewer with adaptive LOD rendering."""

    plot_width = param.Integer(default=950)
    plot_height = param.Integer(default=220)
    plot_width_max = param.Integer(default=950)

    subtract_mean = param.Boolean(default=True)
    normalize_amplitude = param.Boolean(default=False)
    use_datashade = param.Boolean(default=False)
    calibrate = param.Boolean(default=True)
    show_hover = param.Boolean(default=False)

    lod_target_points = param.Integer(default=6_000, bounds=(500, 200_000))
    datashade_threshold = param.Integer(
        default=2_000_000, bounds=(100_000, 100_000_000)
    )

    def __init__(self, use_template=True, **kwargs):
        super().__init__(**kwargs)
        self.use_template = use_template
        self.template = (
            pn.template.BootstrapTemplate(title="MTH5 Viewer V2")
            if use_template
            else None
        )

        self.store = MTDataStore()
        self.raw_run_data = {}
        self.channel_arrays = {}
        self.payloads = {}
        self.channel_colors = {}
        self.subplot_row_assignments = {}
        self._last_refresh_stats = {}

        self.semantic_electric_palette = ["#4477AA", "#66CCEE", "#228833"]
        self.semantic_magnetic_palette = ["#EE6677", "#AA3377", "#CCBB44"]
        self.semantic_aux_palette = ["#BBBBBB", "#999999", "#777777"]
        self.vibrant_palette = cc.glasbey[:20]
        self.viridis_palette = Viridis256

        self._init_widgets()
        self.files_tab = self._make_files_tab()
        self.df_tab = self._make_df_tab()
        self.plots_tab = self._make_plots_tab()
        self.tabs = pn.Tabs(
            ("Files", self.files_tab),
            ("Runs", self.df_tab),
            ("Plots", self.plots_tab),
            dynamic=False,
        )

        self._build_sidebar()

        if self.template is not None:
            self.template.main[:] = [self.tabs]
            self.main_view = self.template
        else:
            self.main_view = pn.Row(
                pn.Card(
                    *self._sidebar_items,
                    title="Controls",
                    width=320,
                    sizing_mode="stretch_height",
                ),
                self.tabs,
                sizing_mode="stretch_width",
            )

    def _init_widgets(self):
        self.cpu_usage = pn.indicators.Number(
            name="CPU", value=0, format="{value}%", width=60
        )
        self.memory_usage = pn.indicators.Number(
            name="Memory", value=0, format="{value}%", width=70
        )
        self.plot_refresh_elapsed = pn.indicators.Number(
            name="Plot Refresh", value=0.0, format="{value:.3f}s", width=95
        )

        self.files = pn.widgets.FileSelector(
            name="Select MTH5 Files",
            directory="~",
            file_pattern="*.h5",
            sizing_mode="stretch_width",
        )
        self.files.param.watch(self._on_files_changed, "value")

        self.runs_table = pn.widgets.Tabulator(
            pd.DataFrame(),
            selectable=True,
            sizing_mode="stretch_both",
            margin=(10, 0, 0, 0),
        )
        self.runs_table.param.watch(self._on_run_selection, "selection")

        self.plot_button = pn.widgets.Button(name="Plot", button_type="primary")
        self.plot_button.on_click(self._on_plot_button)

        self.subtract_mean_checkbox = pn.widgets.Checkbox(
            name="Subtract Mean", value=self.subtract_mean
        )
        self.subtract_mean_checkbox.param.watch(self._on_transform_changed, "value")

        self.normalize_checkbox = pn.widgets.Checkbox(
            name="Normalize Amplitude", value=self.normalize_amplitude
        )
        self.normalize_checkbox.param.watch(self._on_transform_changed, "value")

        self.datashade_checkbox = pn.widgets.Checkbox(
            name="Use Datashade", value=self.use_datashade
        )
        self.datashade_checkbox.param.watch(self._on_render_option_changed, "value")

        self.hover_checkbox = pn.widgets.Checkbox(
            name="Show Hover", value=self.show_hover
        )
        self.hover_checkbox.param.watch(self._on_render_option_changed, "value")

        self.calibrate_checkbox = pn.widgets.Checkbox(
            name="Calibrate", value=self.calibrate
        )
        self.calibrate_checkbox.param.watch(self._on_calibrate_changed, "value")

        self.lod_slider = pn.widgets.IntSlider(
            name="LOD Target Points",
            start=500,
            end=200_000,
            step=500,
            value=self.lod_target_points,
        )
        self.lod_slider.param.watch(self._on_render_option_changed, "value")

        self.datashade_threshold_input = pn.widgets.IntInput(
            name="Datashade Threshold", value=self.datashade_threshold, step=100_000
        )
        self.datashade_threshold_input.param.watch(
            self._on_render_option_changed, "value"
        )

        self.clear_plots_button = pn.widgets.Button(
            name="Clear Plots", button_type="danger"
        )
        self.clear_plots_button.on_click(self._clear_plots)

        self.subplot_row_panel = pn.Column(name="Subplot Row Assignment")

        self.selected_runs = {}

    def _build_sidebar(self):
        self._sidebar_items = [
            self.cpu_usage,
            self.memory_usage,
            self.plot_refresh_elapsed,
            self.calibrate_checkbox,
            self.subtract_mean_checkbox,
            self.normalize_checkbox,
            self.datashade_checkbox,
            self.hover_checkbox,
            self.lod_slider,
            self.datashade_threshold_input,
            self.subplot_row_panel,
            self.plot_button,
            self.clear_plots_button,
        ]
        if self.template is not None:
            self.template.sidebar[:] = self._sidebar_items

    def _make_files_tab(self):
        return pn.Column(self.files, sizing_mode="stretch_width")

    def _make_df_tab(self):
        return pn.Column(self.runs_table, sizing_mode="stretch_width")

    def _make_plots_tab(self):
        self.plot_pane = pn.pane.HoloViews(
            empty_curve(), sizing_mode="stretch_both", max_width=self.plot_width_max
        )
        plots_card = pn.Card(
            self.plot_pane,
            title="Time Series Plots",
            sizing_mode="stretch_width",
            max_width=self.plot_width_max,
        )
        return pn.Column(plots_card, sizing_mode="stretch_width")

    def _on_files_changed(self, event):
        if not event.new:
            return
        run_summary = self.store.load_summaries(event.new)[1]
        self.runs_table.value = run_summary[RUN_SUMMARY_DISPLAY_COLUMNS]
        self.tabs.active = 1

    def _on_run_selection(self, event):
        self.selected_runs = {}
        if len(event.new) == 0:
            return

        for idx in event.new:
            row = self.store.run_summary.iloc[idx]
            self.selected_runs.setdefault(row["file"], []).append(row["hdf5_reference"])

    def _assign_colors(self):
        self.channel_colors = {}
        for idx, key in enumerate(self.channel_arrays.keys()):
            component = key.split(".")[-1].lower()
            if component.startswith("e"):
                color = self.semantic_electric_palette[
                    idx % len(self.semantic_electric_palette)
                ]
            elif component.startswith("h") or component.startswith("b"):
                color = self.semantic_magnetic_palette[
                    idx % len(self.semantic_magnetic_palette)
                ]
            else:
                color = self.semantic_aux_palette[idx % len(self.semantic_aux_palette)]
            self.channel_colors[key] = color

    def _build_default_row_assignments(self):
        keys = list(self.channel_arrays.keys())
        if not self.subplot_row_assignments or set(
            self.subplot_row_assignments.keys()
        ) != set(keys):
            self.subplot_row_assignments = {k: i + 1 for i, k in enumerate(keys)}
        self._update_subplot_row_selectors()

    def _update_subplot_row_selectors(self):
        self.subplot_row_panel.clear()
        keys = list(self.channel_arrays.keys())
        n = len(keys)
        row_options = [str(i + 1) for i in range(max(1, n))]

        for key in keys:
            selector = pn.widgets.Select(
                options=row_options,
                value=str(self.subplot_row_assignments.get(key, 1)),
                width=60,
            )

            def _make_callback(k):
                def _cb(event):
                    self.subplot_row_assignments[k] = int(event.new)
                    self._refresh_plot(reason="layout")

                return _cb

            selector.param.watch(_make_callback(key), "value")
            self.subplot_row_panel.append(
                pn.Row(pn.pane.Markdown(f"**{key}**", width=150), selector)
            )

    def _on_plot_button(self, *events):
        if not self.selected_runs:
            return

        t0 = time.perf_counter()
        self.raw_run_data = self.store.load_selected_runs(
            self.selected_runs, calibrate=self.calibrate_checkbox.value
        )
        self.channel_arrays = dataset_to_channel_arrays(self.raw_run_data)
        self._assign_colors()
        self._build_default_row_assignments()
        self._refresh_plot(reason="data")
        self.plot_refresh_elapsed.value = time.perf_counter() - t0
        self.tabs.active = 2

    def _transform_config(self) -> TransformConfig:
        return TransformConfig(
            subtract_mean=self.subtract_mean_checkbox.value,
            normalize_amplitude=self.normalize_checkbox.value,
        )

    def _render_config(self) -> RenderConfig:
        return RenderConfig(
            plot_height=self.plot_height,
            plot_width=self.plot_width_max,
            lod_target_points=self.lod_slider.value,
            datashade_threshold=self.datashade_threshold_input.value,
            normalize_amplitude=self.normalize_checkbox.value,
            show_hover=self.hover_checkbox.value,
            use_datashade=self.datashade_checkbox.value,
        )

    def _build_row_map(self):
        row_map = {}
        for key, row_idx in self.subplot_row_assignments.items():
            row_map.setdefault(int(row_idx), []).append(key)
        return row_map

    def _refresh_plot(self, reason="values"):
        if not self.channel_arrays:
            self.plot_pane.object = empty_curve()
            return

        t0 = time.perf_counter()
        self.payloads = build_plot_payloads(
            self.channel_arrays, self._transform_config()
        )
        renderer = LODRenderer(color_lookup=self.channel_colors)
        layout = renderer.build_layout(
            self._build_row_map(), self.payloads, self._render_config()
        )
        self.plot_pane.object = layout

        elapsed = time.perf_counter() - t0
        self.plot_refresh_elapsed.value = elapsed
        self._last_refresh_stats = {
            "reason": reason,
            "seconds": elapsed,
            "channels": len(self.payloads),
            "rows": len(set(self.subplot_row_assignments.values())),
            "lod_target": self.lod_slider.value,
            "use_datashade": self.datashade_checkbox.value,
        }

    def _on_transform_changed(self, event):
        self._refresh_plot(reason="transform")

    def _on_render_option_changed(self, event):
        self._refresh_plot(reason="render")

    def _on_calibrate_changed(self, event):
        if self.selected_runs:
            self.raw_run_data = self.store.load_selected_runs(
                self.selected_runs, calibrate=self.calibrate_checkbox.value
            )
            self.channel_arrays = dataset_to_channel_arrays(self.raw_run_data)
            self._assign_colors()
            self._build_default_row_assignments()
            self._refresh_plot(reason="calibrate")

    def _clear_plots(self, event=None):
        self.raw_run_data = {}
        self.channel_arrays = {}
        self.payloads = {}
        self.subplot_row_assignments = {}
        self.plot_pane.object = empty_curve()
        self.subplot_row_panel.clear()

    def view(self):
        return self.main_view


def build_app():
    return MTH5ViewerV2(plot_width=950, plot_height=220).view()


if __name__.startswith("bokeh_app") or __name__ == "__main__":
    panel_app = build_app()
    panel_app.servable()
