import pathlib
import pandas as pd
import panel as pn
import param
import psutil
import xarray
import numpy as np

import holoviews as hv
import hvplot.xarray
from holoviews.operation.datashader import datashade
from holoviews.operation import decimate
import colorcet as cc
from bokeh.palettes import Viridis256

from mth5.mth5 import MTH5
from mth5 import CHANNEL_DTYPE, RUN_SUMMARY_DTYPE

import time

pn.extension("tabulator", sizing_mode="stretch_width")
hv.extension("bokeh")
xarray.set_options(keep_attrs=True)

# --------------------------------------------------------------
# Global Constants
# --------------------------------------------------------------
DATASHADE_THRESHOLD = 5_000_000

CH_SUMMARY_DISPLAY_COLUMNS = [
    "survey",
    "station",
    "run",
    "component",
    "start",
    "end",
    "n_samples",
    "sample_rate",
    "measurement_type",
]

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

COLORMAP = "Magma"

TEMPLATE_KEY = "bootstrap"  # "golden" was default but was not working 17 Apr 2026


def empty_curve():
    return hv.Curve(([], []), kdims=["time"], vdims=["amplitude"]).opts(
        frame_width=1,
        height=1,
        toolbar=None,
    )


def get_templates_dict() -> dict:
    """
    Returns a dictionary of available Panel templates.

    Panel templates are used to define the layout and styling of a Panel application.
    """
    templates = {}
    templates["bootstrap"] = pn.template.BootstrapTemplate
    templates["fast"] = pn.template.FastListTemplate
    templates["golden"] = pn.template.GoldenTemplate
    templates["grid"] = pn.template.FastGridTemplate
    return templates


class MTH5Viewer(param.Parameterized):
    # -------------------------
    # Parameters (reactive state)
    # -------------------------
    plot_width = param.Integer(default=950)
    plot_height = param.Integer(default=450)
    plot_width_max = param.Integer(default=950)

    annotatable = param.Boolean(default=False)
    choose_runs = param.Boolean(
        default=True, doc="True: select runs, False: select channels"
    )
    subtract_mean = param.Boolean(default=True)
    colormap = param.String(default=COLORMAP)

    combine_subplots = param.Boolean(default=True)
    _ordering_version = param.Integer(default=0)
    use_datashade = param.Boolean(
        default=False,
        doc="Enable datashading for rows exceeding the datashade threshold.",
    )
    normalize_amplitude = param.Boolean(
        default=False, doc="Normalize each curve before datashading"
    )

    def __init__(self, use_template=True, **kwargs):
        super().__init__(**kwargs)
        self.use_template = use_template

        # -------------------------
        # Template
        # -------------------------
        self.template = None
        if self.use_template:
            self.template = get_templates_dict()[TEMPLATE_KEY](title="MTH5 Viewer")

        # -------------------------
        # Data state
        # -------------------------
        self.channel_summary = pd.DataFrame(columns=CHANNEL_DTYPE.names)
        self.run_summary = pd.DataFrame(columns=RUN_SUMMARY_DTYPE.names)

        self.selected_channels = {}
        self.selected_runs = {}

        self.data_dict = {}  # key -> xarray object
        self.plot_channel_curves = {}  # key -> pn.pane.HoloViews
        self._curve_data_cache = {}  # key -> lightweight numeric payload for rendering
        self._channel_color_indices = {}  # key -> stable color index
        self._row_render_cache = {}  # cache of rendered row HoloViews objects
        self._layout_cache = {}  # cache of full layout objects by state key
        self._invalidate_cache_next_update = False
        self._force_rescale_next_update = False
        self._curve_pipes = {}  # key -> hv.streams.Pipe for fast in-place updates
        self._curve_dynamicmaps = {}  # key -> hv.DynamicMap backed by Pipe
        self._last_refresh_stats = {}
        self._active_row_order = ()
        self.datashade_cache = {}  # key -> datashaded hv object

        self.subplot_row_assignments = {}  # key -> row index (1-based int)

        # -------------------------
        # Color palettes and maps
        # -------------------------
        # Semantic MT palettes
        self.channel_colors = {}
        self.semantic_electric_palette = ["#4477AA", "#66CCEE", "#228833"]
        self.semantic_magnetic_palette = ["#EE6677", "#AA3377", "#CCBB44"]
        self.semantic_aux_palette = ["#BBBBBB", "#999999", "#777777"]

        # Other palettes
        self.vibrant_palette = cc.glasbey[:20]

        self.viridis_palette = Viridis256

        # Maps for semantic indexing
        self.electric_index_map = {}
        self.magnetic_index_map = {}
        self.aux_index_map = {}

        # -------------------------
        # Widgets
        # -------------------------
        self.cpu_usage = pn.indicators.Number(
            name="CPU",
            value=0,
            format="{value}%",
            colors=[(50, "green"), (75, "orange"), (100, "red")],
            font_size="13pt",
            title_size="8pt",
            width=50,
        )

        self.memory_usage = pn.indicators.Number(
            name="Memory",
            value=0,
            format="{value}%",
            colors=[(50, "green"), (75, "orange"), (100, "red")],
            font_size="13pt",
            title_size="8pt",
            width=50,
        )

        self.plot_refresh_elapsed = pn.indicators.Number(
            name="Plot Refresh",
            value=0.0,
            format="{value:.3f}s",
            font_size="13pt",
            title_size="8pt",
            width=95,
        )

        self.run_or_channel_checkbox = pn.widgets.Checkbox(name="Pick Runs", value=True)
        self.run_or_channel_checkbox.param.watch(self._on_choose_runs_checkbox, "value")

        self.calibrate_checkbox = pn.widgets.Checkbox(name="Calibrate", value=True)
        self.calibrate_checkbox.param.watch(self._on_calibrate_changed, "value")

        self.show_hover_checkbox = pn.widgets.Checkbox(
            name="Show Hover Overlay", value=False
        )
        self.show_hover_checkbox.param.watch(self._on_overlay_hover_changed, "value")

        self.normalize_checkbox = pn.widgets.Checkbox(
            name="Normalize Amplitude", value=False
        )
        self.normalize_checkbox.param.watch(self._on_normalize_changed, "value")

        self.use_datashade_checkbox = pn.widgets.Checkbox(
            name="Use Datashade", value=False
        )
        self.use_datashade_checkbox.param.watch(self._on_use_datashade_changed, "value")

        self.clear_plots_button = pn.widgets.Button(
            name="Clear Plots", button_type="danger"
        )
        self.clear_plots_button.on_click(self.clear_plots)

        self.clear_channels_button = pn.widgets.Button(
            name="Clear Channels", button_type="danger"
        )
        self.clear_channels_button.on_click(self.clear_channels)

        self.plot_button = pn.widgets.Button(name="Plot", button_type="primary")
        self.plot_button.on_click(self._on_plot_button)

        self.subtract_mean_checkbox = pn.widgets.Checkbox(
            name="Subtract Mean", value=True
        )
        self.subtract_mean_checkbox.param.watch(self._on_subtract_mean_changed, "value")

        self.combine_subplots_checkbox = pn.widgets.Checkbox(
            name="Combine Subplots", value=True
        )
        self.combine_subplots_checkbox.param.watch(
            self._on_combine_subplots_changed, "value"
        )

        # Lock color identity (semantic MT colors)
        self.lock_color_identity = pn.widgets.Checkbox(
            name="Lock Color Identity",
            value=False,
        )

        # Palette selector
        self.palette_selector = pn.widgets.Select(
            name="Color Palette",
            options={
                "MT Semantic (Subdued)": "semantic",
                "Vibrant (Glasbey)": "glasbey",
                "Viridis-like": "viridis",
            },
            value="semantic",
        )
        self.palette_selector.param.watch(self._on_palette_changed, "value")
        self.lock_color_identity.param.watch(self._on_palette_changed, "value")

        # Reset ordering button
        self.reset_order_button = pn.widgets.Button(
            name="Reset Ordering",
            button_type="primary",
            width=120,
        )
        self.reset_order_button.on_click(self._reset_ordering)

        # subplot row selectors UI container
        self.subplot_row_panel = pn.Column(name="Subplot Row Assignment")

        # -------------------------
        # Tabs
        # -------------------------
        self.files_tab = self._make_files_tab()
        self.df_tab = self._make_df_tab()
        self.plots_tab = self._make_plots_tab()

        self.tabs = pn.Tabs(
            ("Files", self.files_tab),
            ("DataFrame", self.df_tab),
            ("Plots", self.plots_tab),
            dynamic=False,
        )

        # -------------------------
        # Sidebar
        # -------------------------
        self._build_sidebar()

        # -------------------------
        # Layout
        # -------------------------
        if self.use_template:
            self.template.main[:] = [self.tabs]
            self.main_view = self.template
        else:
            self.main_view = pn.Row(
                pn.Card(
                    *self._sidebar_items,
                    title="Controls",
                    width=300,
                    sizing_mode="stretch_height",
                ),
                self.tabs,
                sizing_mode="stretch_width",
            )

        # -------------------------
        # Resource streaming
        # -------------------------
        self._start_resource_stream()

    # =========================================================
    # Sidebar
    # =========================================================
    def _build_sidebar(self):
        self._sidebar_items = [
            self.cpu_usage,
            self.memory_usage,
            self.plot_refresh_elapsed,
            self.run_or_channel_checkbox,
            self.calibrate_checkbox,
            self.subtract_mean_checkbox,
            self.combine_subplots_checkbox,
            self.normalize_checkbox,
            self.use_datashade_checkbox,
            self.lock_color_identity,
            self.show_hover_checkbox,
            self.palette_selector,
            self.reset_order_button,
            self.subplot_row_panel,
            self.plot_button,
            self.clear_plots_button,
            self.clear_channels_button,
        ]
        if self.template is not None:
            self.template.sidebar[:] = self._sidebar_items

    # =========================================================
    # Tabs
    # =========================================================
    def _make_files_tab(self):
        self.files = pn.widgets.FileSelector(
            name="Select MTH5 Files",
            directory="~",
            file_pattern="*.h5",
            sizing_mode="stretch_width",
        )
        self.files.param.watch(self._on_files_changed, "value")
        return pn.Column(self.files, sizing_mode="stretch_width")

    def _make_df_tab(self):

        df = self.channel_summary.copy()
        df["start"] = pd.to_datetime(df["start"], unit="s").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        df["end"] = pd.to_datetime(df["end"], unit="s").dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

        self.channels_table = pn.widgets.Tabulator(
            df[CH_SUMMARY_DISPLAY_COLUMNS],
            selectable=True,
            sizing_mode="stretch_both",
            margin=(10, 0, 0, 0),
        )

        self.channels_table.param.watch(self._on_table_selection, "selection")
        return pn.Column(self.channels_table, sizing_mode="stretch_width")

    def _make_plots_tab(self):
        self.graphs = pn.Column(
            sizing_mode="stretch_width",
            margin=0,
            max_width=self.plot_width_max,
        )
        self.plot_pane = pn.pane.HoloViews(
            empty_curve(),
            sizing_mode="stretch_both",
            max_width=self.plot_width_max,
        )
        self.plots_card = pn.Card(
            self.plot_pane,
            title="Time Series Plots",
            sizing_mode="stretch_width",
            max_width=self.plot_width_max,
        )
        return pn.Column(self.plots_card, sizing_mode="stretch_width")

    # =========================================================
    # Callbacks / param handlers
    # =========================================================
    def _on_choose_runs_checkbox(self, event):
        self.choose_runs = event.new
        self._refresh_channels_tab()

    def _on_files_changed(self, event):
        self.load_files(event.new)

    def load_files(self, file_paths):
        if not file_paths:
            return
        paths = [str(pathlib.Path(path)) for path in file_paths]
        self._load_summaries_from_files(paths)
        self._refresh_channels_tab()
        self.tabs.active = 0

    def _on_table_selection(self, event):
        self._update_selected_from_table(event.new)

    def _on_plot_button(self, *events):
        self.tabs.active = 2
        self._build_data_dict()
        self._prepare_plot_payloads()
        self._update_subplot_row_selectors()
        self._initialize_plots()

    def _on_calibrate_changed(self, event):
        if self.data_dict:
            self._update_plot_values(reload_data=True)

    def _on_subtract_mean_changed(self, event):
        self.subtract_mean = event.new
        if self.data_dict:
            self._update_plot_values(reload_data=False)

    def _on_combine_subplots_changed(self, event):
        self.combine_subplots = event.new
        self._refresh_plots(reason="layout_changed")

    def _on_palette_changed(self, event=None):
        if self._curve_data_cache:
            self._refresh_plots(reason="style_changed")

    def _on_overlay_hover_changed(self, event):
        self._refresh_plots(reason="style_changed")

    def _on_normalize_changed(self, event):
        self.normalize_amplitude = event.new
        if self._curve_data_cache or self.data_dict:
            # Normalization changes value scale, so force one range recompute pass.
            self._force_rescale_next_update = True
            self._refresh_plots(reason="values_changed")

    def _on_use_datashade_changed(self, event):
        self.use_datashade = event.new
        self._refresh_plots(reason="style_changed")

    # =========================================================
    # Data loading and selection
    # =========================================================
    def _load_summaries_from_files(self, file_paths):
        full_df_channels = pd.DataFrame()
        full_df_runs = pd.DataFrame()

        for file_path in file_paths:
            with MTH5() as m:
                m = m.open_mth5(file_path, mode="r")
                run_df = m.run_summary
                run_df["hdf5_reference"] = run_df["run_hdf5_reference"].apply(
                    lambda ref: m.get_reference_path(ref)
                )
                run_df["file"] = file_path
                run_df.drop(columns=["station_hdf5_reference"], inplace=True)

                channel_df = m.channel_summary.to_dataframe()
                channel_df["hdf5_reference"] = channel_df["hdf5_reference"].apply(
                    lambda ref: m.get_reference_path(ref)
                )
                channel_df["file"] = file_path
                channel_df.drop(
                    columns=["run_hdf5_reference", "station_hdf5_reference"],
                    inplace=True,
                )

            full_df_channels = pd.concat([full_df_channels, channel_df])
            full_df_runs = pd.concat([full_df_runs, run_df])

        self.channel_summary = full_df_channels.reset_index(drop=True)
        self.run_summary = full_df_runs.reset_index(drop=True)

    def _refresh_channels_tab(self):
        if self.choose_runs:
            self.channels_table.value = self.run_summary[RUN_SUMMARY_DISPLAY_COLUMNS]
        else:
            self.channels_table.value = self.channel_summary[CH_SUMMARY_DISPLAY_COLUMNS]

    def _update_selected_from_table(self, selection):
        self.selected_channels = {}
        self.selected_runs = {}
        if not selection:
            return

        if self.choose_runs:
            for idx in selection:
                row = self.run_summary.iloc[idx]
                self.selected_runs.setdefault(row["file"], []).append(
                    row["hdf5_reference"]
                )
        else:
            for idx in selection:
                row = self.channel_summary.iloc[idx]
                self.selected_channels.setdefault(row["file"], []).append(
                    row["hdf5_reference"]
                )

    def _build_data_dict(self):
        """
        Build self.data_dict: key -> xarray object
        """
        t1 = time.perf_counter(), time.process_time()
        out_dict = {}
        self.datashade_cache = {}

        if self.choose_runs:
            for mth5_fn, runs in self.selected_runs.items():
                with MTH5() as m:
                    m.open_mth5(mth5_fn, mode="r")
                    for run_hdf5_path in runs:
                        run = m.from_reference(run_hdf5_path)
                        run_ts = run.to_runts()
                        if self.calibrate_checkbox.value:
                            run_ts.calibrate()
                        data = run_ts.dataset
                        run_key = (
                            f"{run.survey_metadata.id}."
                            f"{run.station_metadata.id}."
                            f"{run.metadata.id}"
                        )
                        out_dict[run_key] = data
        else:
            for mth5_fn, channels in self.selected_channels.items():
                with MTH5() as m:
                    m.open_mth5(mth5_fn, mode="r")
                    for hdf5_path in channels:
                        ch = m.from_reference(hdf5_path)
                        data = ch.to_channel_ts().to_xarray()
                        ch_key = (
                            f"{ch.survey_metadata.id}."
                            f"{ch.station_metadata.id}."
                            f"{ch.run_metadata.id}."
                            f"{ch.metadata.component}"
                        )
                        out_dict[ch_key] = data

        self.data_dict = out_dict
        t2 = time.perf_counter(), time.process_time()
        print(f" Dictionary built in: {t2[0] - t1[0]:.2f} seconds")

    # =========================================================
    # Color logic
    # =========================================================
    def _get_semantic_color(self, ch_key):
        """
        Determine channel color based on MT semantics:
        - Electric: starts with 'e'
        - Magnetic: starts with 'h' or 'b'
        - Auxiliary: anything else
        """
        component = ch_key.split(".")[-1].lower()

        if component.startswith("e"):
            palette = self.semantic_electric_palette
            index = self.electric_index_map.setdefault(
                component, len(self.electric_index_map)
            )
        elif component.startswith("h") or component.startswith("b"):
            palette = self.semantic_magnetic_palette
            index = self.magnetic_index_map.setdefault(
                component, len(self.magnetic_index_map)
            )
        else:
            palette = self.semantic_aux_palette
            index = self.aux_index_map.setdefault(component, len(self.aux_index_map))

        return palette[index % len(palette)]

    def _get_channel_color(self, ch_key, index):
        """
        Returns the color for a channel based on:
        - lock_color_identity checkbox
        - palette_selector choice
        """
        if self.lock_color_identity.value:
            return self._get_semantic_color(ch_key)

        mode = self.palette_selector.value

        if mode == "semantic":
            return self._get_semantic_color(ch_key)

        elif mode == "glasbey":
            return self.vibrant_palette[index % len(self.vibrant_palette)]

        elif mode == "viridis":
            return self.viridis_palette[index % len(self.viridis_palette)]

        return self.vibrant_palette[index % len(self.vibrant_palette)]

    # =========================================================
    # Plotting pipeline
    # =========================================================
    def _prepare_plot_payloads(self):
        """
        Build per-channel numeric payloads from self.data_dict.
        This stage is pure data preparation and does not render plots.
        """
        t1 = time.perf_counter(), time.process_time()
        self._curve_data_cache = {}
        self._channel_color_indices = {}
        self._row_render_cache = {}
        self._layout_cache = {}
        keys = list(self.data_dict.keys())

        for idx, key in enumerate(keys):
            data = self.data_dict[key]

            if self.subtract_mean:
                if isinstance(data, xarray.DataArray):
                    data = data - data.mean()
                elif isinstance(data, xarray.Dataset):
                    data = data - data.mean()

            if self.choose_runs and isinstance(data, xarray.Dataset):
                for j, ch in enumerate(data.data_vars):
                    ch_da = data[ch]
                    ch_key = f"{key}.{ch_da.component}"
                    color_index = len(self._channel_color_indices)
                    self._channel_color_indices[ch_key] = color_index
                    self._curve_data_cache[ch_key] = self._build_curve_payload(
                        ch_da, color_index
                    )
            else:
                color_index = len(self._channel_color_indices)
                self._channel_color_indices[key] = color_index
                self._curve_data_cache[key] = self._build_curve_payload(
                    data, color_index
                )

        self._refresh_channel_colors()
        self._init_row_assignments()
        self._sync_stream_curves()
        t2 = time.perf_counter(), time.process_time()
        print(f" Plots generated in: {t2[0] - t1[0]:.2f} seconds")

    def _build_or_update_plots(self):
        """Backward-compatible wrapper for existing callers."""
        self._prepare_plot_payloads()

    def _refresh_channel_colors(self):
        self.channel_colors = {}
        for ch_key, color_index in self._channel_color_indices.items():
            self.channel_colors[ch_key] = self._get_channel_color(ch_key, color_index)

    def _sync_stream_curves(self):
        active_keys = set(self._curve_data_cache.keys())

        for key in list(self._curve_pipes.keys()):
            if key not in active_keys:
                self._curve_pipes.pop(key, None)
                self._curve_dynamicmaps.pop(key, None)

        for key in active_keys:
            if key not in self._curve_pipes or key not in self._curve_dynamicmaps:
                self._create_stream_curve(key)

    def _create_stream_curve(self, ch_key):
        payload = self._curve_data_cache[ch_key]
        y = payload["y_normalized"] if self.normalize_amplitude else payload["y"]
        pipe = hv.streams.Pipe(data=(payload["x"], y))

        def _curve_from_stream(data, _key=ch_key):
            active_payload = self._curve_data_cache.get(_key)
            if active_payload is None:
                return empty_curve()

            if data is None:
                ys = (
                    active_payload["y_normalized"]
                    if self.normalize_amplitude
                    else active_payload["y"]
                )
                data = (active_payload["x"], ys)

            return hv.Curve(
                data,
                kdims=[active_payload["dim"]],
                vdims=[active_payload["vdim"]],
            ).opts(
                height=self.plot_height,
                ylabel=active_payload["units"],
                title=_key,
                color=self.channel_colors.get(_key, "#4477AA"),
                tools=["hover"] if self.show_hover_checkbox.value else [],
                show_grid=True,
                gridstyle={"grid_line_color": "lightgray", "grid_line_alpha": 0.5},
                xticks=20,
            )

        self._curve_pipes[ch_key] = pipe
        self._curve_dynamicmaps[ch_key] = hv.DynamicMap(
            _curve_from_stream, streams=[pipe]
        )

    def _fast_stream_value_update(self):
        if not self._curve_data_cache:
            return False
        if self.use_datashade:
            return False

        self._sync_stream_curves()
        if not self._curve_pipes:
            return False

        for key, payload in self._curve_data_cache.items():
            pipe = self._curve_pipes.get(key)
            if pipe is None:
                continue
            ys = payload["y_normalized"] if self.normalize_amplitude else payload["y"]
            pipe.send((payload["x"], ys))

        return True

    def _build_curve_payload(self, ch_data, color_index):
        dim = list(ch_data.dims)[0]
        y = np.asarray(ch_data.values)
        y_min = float(y.min()) if y.size else 0.0
        y_ptp = float(np.ptp(y)) if y.size else 0.0
        return {
            "x": np.asarray(ch_data[dim].values),
            "y": y,
            "y_normalized": ((y - y_min) / y_ptp) if y_ptp > 0 else (y * 0),
            "dim": dim,
            "vdim": ch_data.name or "amplitude",
            "units": getattr(ch_data, "units", ""),
            "color_index": color_index,
            "n_points": int(y.size),
        }

    def _make_channel_curve(self, ch_key, payload):
        """
        Return a pure HoloViews Curve (no datashader, no Pane).
        Used for row-level overlays and datashading.
        """

        color = self._get_channel_color(ch_key, payload["color_index"])
        self.channel_colors[ch_key] = color

        curve = hv.Curve(
            (payload["x"], payload["y"]),
            kdims=[payload["dim"]],
            vdims=[payload["vdim"]],
        ).opts(
            height=self.plot_height,
            ylabel=payload["units"],
            title=ch_key,
            color=color,
            tools=["hover"] if self.show_hover_checkbox.value else [],
            show_grid=True,
            gridstyle={"grid_line_color": "lightgray", "grid_line_alpha": 0.5},
            xticks=20,
        )

        return curve

    def _init_row_assignments(self):
        keys = list(self._curve_data_cache.keys())
        if not self.subplot_row_assignments or set(
            self.subplot_row_assignments.keys()
        ) != set(keys):
            self.subplot_row_assignments = {key: i + 1 for i, key in enumerate(keys)}
            self._ordering_version += 1

    def _update_subplot_row_selectors(self):
        self.subplot_row_panel.clear()
        keys = list(self._curve_data_cache.keys())
        n = len(keys)
        row_options = [str(i + 1) for i in range(n)]

        for key in keys:
            current_row = self.subplot_row_assignments.get(key, 1)
            selector = pn.widgets.Select(
                options=row_options,
                value=str(current_row),
                width=60,
            )

            def _make_callback(k):
                def _cb(event):
                    self.subplot_row_assignments[k] = int(event.new)
                    self._ordering_version += 1
                    self._refresh_plots(reason="layout_changed")

                return _cb

            selector.param.watch(_make_callback(key), "value")
            row = pn.Row(pn.pane.Markdown(f"**{key}**", width=150), selector)
            self.subplot_row_panel.append(row)

    def _reset_ordering(self, event=None):
        keys = list(self._curve_data_cache.keys())
        self.subplot_row_assignments = {k: i + 1 for i, k in enumerate(keys)}
        self._ordering_version += 1
        self._update_subplot_row_selectors()
        self._refresh_plots(reason="layout_changed")

    def _get_length(self, key):
        try:
            data = self.data_dict[key.rsplit(".", 1)[0]]
        except KeyError:
            data = self.data_dict[key]
        if isinstance(data, xarray.DataArray):
            return len(data)
        elif isinstance(data, xarray.Dataset):
            return data.sizes["time"]
        return 0

    def _get_row_cache_key(self, keys):
        return (
            tuple(keys),
            self.normalize_amplitude,
            self.use_datashade,
            self.palette_selector.value,
            self.lock_color_identity.value,
            self.plot_width,
            self.plot_height,
        )

    def _get_layout_cache_key(self):
        row_items = tuple(
            sorted((k, int(v)) for k, v in self.subplot_row_assignments.items())
        )
        return (
            row_items,
            self.normalize_amplitude,
            self.use_datashade,
            self.palette_selector.value,
            self.lock_color_identity.value,
            self.show_hover_checkbox.value,
            self.plot_width,
            self.plot_height,
        )

    def _build_row_plot(self, row_idx, keys):
        cache_key = self._get_row_cache_key(keys)
        cached_final = self._row_render_cache.get(cache_key)
        if cached_final is not None:
            return cached_final

        use_datashader = self.use_datashade and any(
            self._curve_data_cache[k].get("n_points", self._get_length(k))
            > DATASHADE_THRESHOLD
            for k in keys
        )

        if use_datashader:
            hv_objs = {}
            for k in keys:
                payload = self._curve_data_cache[k]
                xs = payload["x"]
                ys = (
                    payload["y_normalized"]
                    if self.normalize_amplitude
                    else payload["y"]
                )

                unified = hv.Curve(
                    (xs, ys), kdims=[payload["dim"]], vdims=["amplitude"]
                ).opts(
                    color=self.channel_colors[k],
                    title=k,
                )
                hv_objs[k] = unified

            if not hv_objs:
                return empty_curve()

            overlay_raw = hv.NdOverlay(hv_objs, kdims="channel")
        else:
            overlays = []
            for k in keys:
                stream_curve = self._curve_dynamicmaps.get(k)
                if stream_curve is None:
                    self._create_stream_curve(k)
                    stream_curve = self._curve_dynamicmaps.get(k)
                if stream_curve is None:
                    continue
                overlays.append(stream_curve.relabel(k))

            if not overlays:
                return empty_curve()

            overlay_raw = overlays[0]
            for stream_overlay in overlays[1:]:
                overlay_raw = overlay_raw * stream_overlay

        if use_datashader:
            print(f"Datashading row {row_idx} with channels: {keys}")
            color_key = {k: self.channel_colors[k] for k in keys}
            print(color_key)

            final = datashade(
                overlay_raw,
                aggregator="any",
                height=self.plot_height,
                color_key=color_key,
            )

            if self.show_hover_checkbox.value:
                print("Adding a hover overlay does not work yet. Skipping.")
        else:
            final = overlay_raw

        final = final.opts(
            frame_width=min(self.plot_width, self.plot_width_max),
            framewise=True,
            axiswise=True,
        )
        self._row_render_cache[cache_key] = final
        return final

    def _compose_layout(self):
        layout_cache_key = self._get_layout_cache_key()
        cached_layout = self._layout_cache.get(layout_cache_key)
        if cached_layout is not None:
            return cached_layout, self._active_row_order

        row_map = {}
        for key, row_idx in self.subplot_row_assignments.items():
            row_map.setdefault(row_idx, []).append(key)

        sorted_rows = sorted(row_map.keys())
        row_plots = []
        for row_idx in sorted_rows:
            keys = row_map[row_idx]
            if not keys:
                continue
            row_plot = self.update_plot(row_idx, keys=keys)
            if row_plot is not None:
                row_plots.append(row_plot)

        row_order = tuple(sorted_rows)
        if row_plots:
            layout = hv.Layout(row_plots).cols(1).opts(shared_axes=True)
        else:
            layout = empty_curve()

        self._layout_cache[layout_cache_key] = layout

        return layout, row_order

    def update_plot(self, row_idx, keys=None):
        if keys is None:
            keys = [
                key
                for key, assigned_row in self.subplot_row_assignments.items()
                if assigned_row == row_idx
            ]

        if not keys:
            return None

        return self._build_row_plot(row_idx, keys)

    def _initialize_plots(self):
        """
        Initialize plot layout after payloads are prepared.
        Called on first plot render or structural data changes.
        """
        if not self._curve_data_cache:
            self.plot_pane.object = empty_curve()
            self._active_row_order = ()
            return

        self._invalidate_plot_caches()
        layout, row_order = self._compose_layout()
        self.plot_pane.object = layout
        self._active_row_order = row_order

    def update_plots(self):
        """
        Lightweight plot refresh using existing payloads and row assignments.
        """
        if not self._curve_data_cache:
            self.plot_pane.object = empty_curve()
            self._active_row_order = ()
            return

        if self._invalidate_cache_next_update:
            self._invalidate_plot_caches()
            self._invalidate_cache_next_update = False

        layout, row_order = self._compose_layout()
        self.plot_pane.object = layout
        self._active_row_order = row_order

    def _invalidate_plot_caches(self):
        self._row_render_cache = {}
        self._layout_cache = {}

    def _update_plot_values(self, reload_data=False):
        """
        Update value payloads used by curves without rebuilding UI controls.

        reload_data=True is used for calibrate changes because values come
        from a different source representation.
        """
        previous_assignments = dict(self.subplot_row_assignments)

        if reload_data:
            self._build_data_dict()

        self._prepare_plot_payloads()

        if previous_assignments:
            keys = set(self._curve_data_cache.keys())
            self.subplot_row_assignments = {
                k: r for k, r in previous_assignments.items() if k in keys
            }
            missing = [
                k
                for k in self._curve_data_cache.keys()
                if k not in self.subplot_row_assignments
            ]
            if missing:
                start_row = max(self.subplot_row_assignments.values(), default=0)
                for i, k in enumerate(missing, start=1):
                    self.subplot_row_assignments[k] = start_row + i

        keys_changed = set(previous_assignments.keys()) != set(
            self._curve_data_cache.keys()
        )

        if keys_changed:
            self._update_subplot_row_selectors()
            self._invalidate_plot_caches()
            self.update_plots()
            return

        self._refresh_plots(reason="values_changed")

    def _record_refresh_timing(self, reason, mode, start_time):
        elapsed = time.perf_counter() - start_time
        self.plot_refresh_elapsed.value = elapsed
        self._last_refresh_stats = {
            "reason": reason,
            "mode": mode,
            "seconds": elapsed,
            "channels": len(self._curve_data_cache),
            "rows": len({int(v) for v in self.subplot_row_assignments.values()}),
        }
        print(
            f" [plot-refresh] reason={reason} mode={mode} elapsed={elapsed:.3f}s "
            f"channels={self._last_refresh_stats['channels']} rows={self._last_refresh_stats['rows']}"
        )

    def _refresh_plots(self, reason="values_changed"):
        """
        Route plotting updates by intent.
        - data_changed: data selection changed, rebuild payloads and initialize
        - values_changed: update y values (subtract mean/normalize/calibrate)
        - style_changed: update colors/tools/datashade state
        - layout_changed: row assignments or combine mode changed
        """
        t0 = time.perf_counter()

        if reason == "data_changed":
            self._prepare_plot_payloads()
            self._initialize_plots()
            self._record_refresh_timing(reason, "reinitialize", t0)
            return

        if reason == "values_changed":
            if self._fast_stream_value_update():
                if self._force_rescale_next_update:
                    self._force_rescale_next_update = False
                    self._invalidate_cache_next_update = True
                    self.update_plots()
                    self._record_refresh_timing(reason, "stream-update+rescale", t0)
                    return
                self._record_refresh_timing(reason, "stream-update", t0)
                return
            self._invalidate_cache_next_update = False
            self.update_plots()
            self._force_rescale_next_update = False
            self._record_refresh_timing(reason, "layout-cache", t0)
            return

        if reason == "style_changed":
            self._refresh_channel_colors()
            self._fast_stream_value_update()
            self._invalidate_cache_next_update = True
            self.update_plots()
            self._record_refresh_timing(reason, "style-rebuild", t0)
            return

        if reason == "layout_changed":
            self._invalidate_cache_next_update = True
            self.update_plots()
            self._record_refresh_timing(reason, "layout-rebuild", t0)
            return

        self._invalidate_cache_next_update = True
        self.update_plots()
        self._record_refresh_timing(reason, "fallback-rebuild", t0)

    def _render_plots(self):
        self._refresh_plots(reason="values_changed")

    # =========================================================
    # Clear / reset
    # =========================================================
    def clear_plots(self, event=None):
        self.data_dict = {}
        self.plot_channel_curves = {}
        self._curve_data_cache = {}
        self._channel_color_indices = {}
        self._row_render_cache = {}
        self._layout_cache = {}
        self._curve_pipes = {}
        self._curve_dynamicmaps = {}
        self._active_row_order = ()
        self.datashade_cache = {}
        self.subplot_row_assignments = {}
        self.plot_pane.object = empty_curve()
        self.subplot_row_panel.clear()

    def clear_channels(self, event=None):
        self.selected_channels = {}
        self.selected_runs = {}
        self.channel_summary = pd.DataFrame(columns=CHANNEL_DTYPE.names)
        self.run_summary = pd.DataFrame(columns=RUN_SUMMARY_DTYPE.names)
        self._refresh_channels_tab()

    # =========================================================
    # Resource Streaming
    # =========================================================
    def _start_resource_stream(self):
        def update_resources():
            mem = psutil.virtual_memory().percent
            cpu = psutil.cpu_percent()
            self.cpu_usage.value = cpu
            self.memory_usage.value = mem

        pn.state.add_periodic_callback(update_resources, period=1000)

    # =========================================================
    # Entry Point
    # =========================================================
    def view(self):
        return self.main_view


def build_app():
    mth5_viewer = MTH5Viewer(plot_width=700, plot_height=200)
    return mth5_viewer.view()


if __name__.startswith("bokeh_app") or __name__ == "__main__":
    panel_app = build_app()
    panel_app.servable()
