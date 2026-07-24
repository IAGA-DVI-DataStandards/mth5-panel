from __future__ import annotations

from pathlib import Path

import pandas as pd

from mth5_panel.make_mth5 import MakeMTH5PanelApp


def test_update_input_type_for_client_sets_allowed_choice():
    app = MakeMTH5PanelApp()
    app.client_type = "FDSN Client"
    app._update_input_type_for_client()

    assert app.input_type == "Request CSV"
    assert app.param.input_type.objects == ["Request CSV"]


def test_split_paths_parses_commas_and_newlines():
    raw = "a.mseed, b.mseed\n c.mseed"

    result = MakeMTH5PanelApp._split_paths(raw)

    assert result == ["a.mseed", "b.mseed", "c.mseed"]


def test_common_kwargs_maps_none_compression():
    app = MakeMTH5PanelApp()
    app.h5_compression = "none"

    kwargs = app._common_kwargs()

    assert kwargs["h5_compression"] is None
    assert kwargs["mth5_file_mode"] == "w"


def test_run_create_fdsn_dispatches_with_csv(monkeypatch, tmp_path):
    app = MakeMTH5PanelApp()
    app.client_type = "FDSN Client"
    app.request_csv = str(tmp_path / "request.csv")
    app.fdsn_client = "IRIS"
    app.save_path = str(tmp_path)
    app.mth5_filename = "fdsn_test.h5"

    request_df = pd.DataFrame(
        {
            "network": ["8P"],
            "station": ["CAS04"],
            "location": [""],
            "channel": ["LFE"],
            "start": ["2020-01-01T00:00:00"],
            "end": ["2020-01-02T00:00:00"],
        }
    )
    request_df.to_csv(app.request_csv, index=False)

    call_args: dict[str, object] = {}

    def _fake_from_fdsn_client(request_df_arg, **kwargs):
        call_args["request_df"] = request_df_arg
        call_args["kwargs"] = kwargs
        return tmp_path / "out_fdsn.h5"

    monkeypatch.setattr(
        "mth5_panel.make_mth5.MakeMTH5.from_fdsn_client", _fake_from_fdsn_client
    )

    app._run_create()

    assert "Created MTH5 successfully" in app.status
    assert app.created_mth5_path == str(tmp_path / "out_fdsn.h5")
    assert isinstance(call_args["request_df"], pd.DataFrame)
    assert call_args["kwargs"]["client"] == "IRIS"
    assert call_args["kwargs"]["mth5_filename"] == "fdsn_test.h5"


def test_run_create_metronix_dispatches(monkeypatch, tmp_path):
    app = MakeMTH5PanelApp()
    app.client_type = "Metronix"
    app.data_path = str(tmp_path)
    app.save_path = str(tmp_path)
    app.sample_rates = "128,512"
    app.run_name_zeros = 3

    call_args: dict[str, object] = {}

    def _fake_from_metronix(data_path_arg, **kwargs):
        call_args["data_path"] = data_path_arg
        call_args["kwargs"] = kwargs
        return Path(tmp_path) / "out_metronix.h5"

    monkeypatch.setattr(
        "mth5_panel.make_mth5.MakeMTH5.from_metronix", _fake_from_metronix
    )

    app._run_create()

    assert "Created MTH5 successfully" in app.status
    assert app.created_mth5_path == str(Path(tmp_path) / "out_metronix.h5")
    assert call_args["data_path"] == str(tmp_path)
    assert call_args["kwargs"]["sample_rates"] == [128.0, 512.0]
    assert call_args["kwargs"]["run_name_zeros"] == 3


def test_run_create_handles_invalid_extra_kwargs_json(tmp_path):
    app = MakeMTH5PanelApp()
    app.client_type = "Metronix"
    app.data_path = str(tmp_path)
    app.extra_kwargs_json = "[1, 2, 3]"

    app._run_create()

    assert app.status.startswith("Error:")
    assert app.created_mth5_path == ""


def test_run_create_requires_stationxml_for_fdsn_inventory(tmp_path):
    app = MakeMTH5PanelApp()
    app.client_type = "FDSN StationXML + miniSEED"
    app.save_path = str(tmp_path)
    app.miniseed_files = str(tmp_path / "run.mseed")

    app._run_create()

    assert app.status.startswith("Error:")
