from __future__ import annotations

from pathlib import Path

import numpy as np
import panel as pn
import pytest

from mth5_panel.make_mth5 import MakeMTH5PanelApp
from mth5_panel.mth5_panel_app import MTH5MasterPanelApp


class DummyTabs:
    def __init__(self):
        self.active = 0


class DummyViewer:
    def __init__(self):
        self.loaded: list[list[str]] = []
        self.tabs = DummyTabs()

    def load_files(self, file_paths):
        self.loaded.append(list(file_paths))

    def view(self):
        return pn.Column("viewer")


def test_master_handoff_enabled_after_create_path_update():
    make_app = MakeMTH5PanelApp()
    viewer = DummyViewer()
    app = MTH5MasterPanelApp(make_app=make_app, viewer_app=viewer)

    assert app._handoff_button.disabled is True

    make_app.created_mth5_path = "example.h5"

    assert app._handoff_button.disabled is False
    assert app._created_path_display.value == "example.h5"


def test_master_handoff_missing_file_sets_error(tmp_path):
    make_app = MakeMTH5PanelApp()
    viewer = DummyViewer()
    app = MTH5MasterPanelApp(make_app=make_app, viewer_app=viewer)

    missing_file = tmp_path / "missing.h5"
    make_app.created_mth5_path = str(missing_file)

    app._open_created_in_viewer()

    assert app.status.startswith("Error:")
    assert viewer.loaded == []


def test_master_handoff_loads_viewer_and_switches_tabs(tmp_path):
    make_app = MakeMTH5PanelApp()
    viewer = DummyViewer()
    app = MTH5MasterPanelApp(make_app=make_app, viewer_app=viewer)

    output_file = tmp_path / "created.h5"
    output_file.write_text("placeholder")
    make_app.created_mth5_path = str(output_file)

    app._open_created_in_viewer()

    assert viewer.loaded == [[str(output_file)]]
    assert viewer.tabs.active == 1
    assert app.tabs.active == 1
    assert app.status.startswith("Loaded")


def test_viewer_load_files_calls_refresh(monkeypatch, tmp_path):
    pytest.importorskip("holoviews")
    from mth5_panel.mth5_viewer import MTH5Viewer

    viewer = MTH5Viewer(use_template=False)
    input_file = tmp_path / "input.h5"

    calls: dict[str, object] = {}

    def _fake_load(paths):
        calls["paths"] = paths

    def _fake_refresh():
        calls["refreshed"] = True

    monkeypatch.setattr(viewer, "_load_summaries_from_files", _fake_load)
    monkeypatch.setattr(viewer, "_refresh_channels_tab", _fake_refresh)

    viewer.load_files([str(input_file)])

    assert calls["paths"] == [str(Path(input_file))]
    assert calls["refreshed"] is True
    assert viewer.tabs.active == 0


def test_viewer_use_datashade_toggle_uses_update_plots(monkeypatch):
    pytest.importorskip("holoviews")
    from mth5_panel.mth5_viewer import MTH5Viewer

    viewer = MTH5Viewer(use_template=False)

    calls = {"update": 0, "render": 0}

    def _fake_update():
        calls["update"] += 1

    def _fake_render():
        calls["render"] += 1

    monkeypatch.setattr(viewer, "update_plots", _fake_update)
    monkeypatch.setattr(viewer, "_render_plots", _fake_render)

    viewer.use_datashade_checkbox.value = True

    assert viewer.use_datashade is True
    assert calls["update"] == 1
    assert calls["render"] == 0


def test_viewer_normalize_callback_uses_update_plots(monkeypatch):
    pytest.importorskip("holoviews")
    from mth5_panel.mth5_viewer import MTH5Viewer

    viewer = MTH5Viewer(use_template=False)
    viewer.data_dict = {"dummy": object()}

    calls = {"update": 0, "render": 0}

    def _fake_update():
        calls["update"] += 1

    def _fake_render():
        calls["render"] += 1

    monkeypatch.setattr(viewer, "update_plots", _fake_update)
    monkeypatch.setattr(viewer, "_render_plots", _fake_render)

    viewer._on_normalize_changed(type("Event", (), {"new": True})())

    assert viewer.normalize_amplitude is True
    assert calls["update"] == 1
    assert calls["render"] == 0


def test_viewer_calibrate_callback_updates_values(monkeypatch):
    pytest.importorskip("holoviews")
    from mth5_panel.mth5_viewer import MTH5Viewer

    viewer = MTH5Viewer(use_template=False)
    viewer.data_dict = {"dummy": object()}

    calls = {"reload_data": None}

    def _fake_update_plot_values(reload_data=False):
        calls["reload_data"] = reload_data

    monkeypatch.setattr(viewer, "_update_plot_values", _fake_update_plot_values)

    viewer._on_calibrate_changed(type("Event", (), {"new": False})())

    assert calls["reload_data"] is True


def test_viewer_datashade_respects_toggle(monkeypatch):
    pytest.importorskip("holoviews")
    from mth5_panel import mth5_viewer as viewer_mod

    viewer = viewer_mod.MTH5Viewer(use_template=False)
    key = "survey.station.run.ex"

    viewer._curve_data_cache = {
        key: {
            "x": np.asarray([0, 1]),
            "y": np.asarray([0, 1]),
            "dim": "time",
            "vdim": "amplitude",
            "units": "",
            "color_index": 0,
        }
    }
    viewer._channel_color_indices = {key: 0}
    viewer.subplot_row_assignments = {key: 1}
    viewer.channel_colors = {key: "#4477AA"}

    monkeypatch.setattr(
        viewer, "_get_length", lambda _k: viewer_mod.DATASHADE_THRESHOLD + 1
    )

    calls = {"datashade": 0}

    def _fake_datashade(*args, **kwargs):
        calls["datashade"] += 1
        return args[0]

    monkeypatch.setattr(viewer_mod, "datashade", _fake_datashade)

    viewer.use_datashade = False
    viewer._render_plots()
    assert calls["datashade"] == 0

    viewer.use_datashade = True
    viewer._render_plots()
    assert calls["datashade"] == 1


def test_viewer_update_plot_returns_row_plot():
    pytest.importorskip("holoviews")
    from mth5_panel.mth5_viewer import MTH5Viewer

    viewer = MTH5Viewer(use_template=False)
    key = "survey.station.run.ex"
    viewer.channel_colors = {key: "#4477AA"}
    viewer._curve_data_cache = {
        key: {
            "x": np.asarray([0, 1]),
            "y": np.asarray([0.0, 2.0]),
            "y_normalized": np.asarray([0.0, 1.0]),
            "dim": "time",
            "vdim": "amplitude",
            "units": "",
            "color_index": 0,
            "n_points": 2,
        }
    }
    viewer.subplot_row_assignments = {key: 1}

    row_plot = viewer.update_plot(1, keys=[key])

    assert row_plot is not None


def test_viewer_update_plots_updates_shared_plot_pane():
    pytest.importorskip("holoviews")
    import holoviews as hv
    from mth5_panel.mth5_viewer import MTH5Viewer

    viewer = MTH5Viewer(use_template=False)
    key = "survey.station.run.ex"
    viewer._curve_data_cache = {
        key: {
            "x": np.asarray([0, 1]),
            "y": np.asarray([0.0, 2.0]),
            "y_normalized": np.asarray([0.0, 1.0]),
            "dim": "time",
            "vdim": "amplitude",
            "units": "",
            "color_index": 0,
            "n_points": 2,
        }
    }
    viewer.channel_colors = {key: "#4477AA"}
    viewer.subplot_row_assignments = {key: 1}

    viewer.update_plots()
    first_object = viewer.plot_pane.object

    viewer.normalize_amplitude = True
    viewer.update_plots()

    assert viewer.plot_pane.object is not None
    assert viewer.plot_pane is not None
    assert viewer.plot_pane.object is not first_object
    assert isinstance(viewer.plot_pane.object, hv.Layout)
