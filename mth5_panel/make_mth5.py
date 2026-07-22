# -*- coding: utf-8 -*-
"""Panel application for building MTH5 files with MakeMTH5.

Run with:

    panel serve mth5/clients/make_mth5_panel.py --show
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import param

from mth5.clients.make_mth5 import MakeMTH5

try:
    import panel as pn
except ImportError as error:  # pragma: no cover - only raised in missing optional dep envs
    raise ImportError(
        "panel and param are required for the MakeMTH5 panel app. "
        "Install with `pip install panel param`."
    ) from error


pn.extension(sizing_mode="stretch_width")


class MakeMTH5PanelApp(param.Parameterized):
    """Interactive UI for creating MTH5 files from supported clients."""

    client_type = param.Selector(
        default="Zen",
        objects=[
            "FDSN Client",
            "FDSN StationXML + miniSEED",
            "USGS Geomag",
            "INTERMAG",
            "Zen",
            "Phoenix",
            "LEMI",
            "LEMI-424",
            "LEMI-417",
            "Metronix",
            "NIMS",
            "UoA",
        ],
        doc="Client/data source to build from.",
    )

    input_type = param.Selector(
        default="Directory", objects=["Directory", "Request CSV", "Files"]
    )

    data_path = param.String(default=str(Path.cwd()), doc="Input data directory path.")
    save_path = param.String(default=str(Path.cwd()), doc="Output directory path.")
    mth5_filename = param.String(default="from_client.h5", doc="Output MTH5 filename.")

    mth5_version = param.Selector(default="0.2.0", objects=["0.2.0", "0.1.0"])
    mth5_file_mode = param.Selector(default="w", objects=["w", "a"])
    interact = param.Boolean(default=False)

    h5_compression = param.Selector(
        default="gzip", objects=["gzip", "lzf", "szip", "none"]
    )
    h5_compression_opts = param.Integer(default=4, bounds=(0, 9))
    h5_shuffle = param.Boolean(default=True)
    h5_fletcher32 = param.Boolean(default=True)
    h5_data_level = param.Integer(default=1, bounds=(0, 5))

    request_csv = param.String(default="", doc="Path to request CSV for FDSN/Geomag/INTERMAG.")
    station_xml_path = param.String(default="", doc="Path to StationXML file.")
    miniseed_files = param.String(
        default="", doc="miniSEED file paths (comma or newline separated)."
    )

    sample_rates = param.String(default="4096,1024,256")
    calibration_path = param.String(default="")
    survey_id = param.String(default="")
    station_id = param.String(default="")
    run_id = param.String(default="001")
    combine = param.Boolean(default=True)

    fdsn_client = param.String(default="IRIS")
    receiver_calibration_path = param.String(default="")
    sensor_calibration_path = param.String(default="")
    run_name_zeros = param.Integer(default=0, bounds=(0, 6))

    instrument_type = param.Selector(default="pr624", objects=["pr624", "orange"])
    extra_kwargs_json = param.String(
        default="{}",
        doc="Extra kwargs as JSON. Example: {\"sample_rate\": 10.0, \"dipole_length_ex\": 50.0}",
    )

    status = param.String(default="Ready")

    run = param.Action(lambda self: self._run_create(), label="Create MTH5")

    _INPUT_TYPE_BY_CLIENT = {
        "FDSN Client": ["Request CSV"],
        "FDSN StationXML + miniSEED": ["Files"],
        "USGS Geomag": ["Request CSV"],
        "INTERMAG": ["Request CSV"],
        "Zen": ["Directory"],
        "Phoenix": ["Directory"],
        "LEMI": ["Directory"],
        "LEMI-424": ["Directory"],
        "LEMI-417": ["Directory"],
        "Metronix": ["Directory"],
        "NIMS": ["Directory"],
        "UoA": ["Directory"],
    }

    _FILE_TYPE_HELP = {
        "FDSN Client": "Request CSV with columns: network, station, location, channel, start, end.",
        "FDSN StationXML + miniSEED": "StationXML file and one or more miniSEED files.",
        "USGS Geomag": "Request CSV with geomag columns: observatory, type, elements, sampling_period, start, end.",
        "INTERMAG": "Request CSV with INTERMAG request columns.",
        "Zen": "Directory containing Z3D files and optional amtant.cal calibration file.",
        "Phoenix": "Directory containing Phoenix station folders/files and optional rxcal/scal calibration directories.",
        "LEMI": "Directory containing LEMI files (.txt or .B423).",
        "LEMI-424": "Directory containing LEMI-424 files (.txt).",
        "LEMI-417": "Directory containing LEMI-417 files.",
        "Metronix": "Survey or station directory containing Metronix ATSS + JSON structure.",
        "NIMS": "Directory containing NIMS .BIN files and optional calibration file.",
        "UoA": "Directory or file for UoA PR6-24 or Orange Box data.",
    }

    def __init__(self, **params: Any):
        super().__init__(**params)

        self._browser = pn.widgets.FileSelector(str(Path.cwd()))
        self._set_data_button = pn.widgets.Button(name="Use Selected as Data Path")
        self._set_save_button = pn.widgets.Button(name="Use Selected as Save Path")
        self._set_request_button = pn.widgets.Button(name="Use Selected as Request CSV")
        self._set_stationxml_button = pn.widgets.Button(name="Use Selected as StationXML")
        self._set_miniseed_button = pn.widgets.Button(name="Add Selected to miniSEED List")

        self._set_data_button.on_click(self._set_data_from_selection)
        self._set_save_button.on_click(self._set_save_from_selection)
        self._set_request_button.on_click(self._set_request_from_selection)
        self._set_stationxml_button.on_click(self._set_stationxml_from_selection)
        self._set_miniseed_button.on_click(self._append_miniseed_from_selection)

        self.param.watch(self._update_input_type_for_client, "client_type")
        self._update_input_type_for_client()

    def _selected_path(self) -> Path | None:
        if not self._browser.value:
            return None
        return Path(self._browser.value[0])

    def _set_data_from_selection(self, _event=None):
        selected = self._selected_path()
        if selected is None:
            self.status = "No selection made in file browser."
            return
        self.data_path = str(selected if selected.is_dir() else selected.parent)

    def _set_save_from_selection(self, _event=None):
        selected = self._selected_path()
        if selected is None:
            self.status = "No selection made in file browser."
            return
        self.save_path = str(selected if selected.is_dir() else selected.parent)

    def _set_request_from_selection(self, _event=None):
        selected = self._selected_path()
        if selected is None:
            self.status = "No selection made in file browser."
            return
        self.request_csv = str(selected)

    def _set_stationxml_from_selection(self, _event=None):
        selected = self._selected_path()
        if selected is None:
            self.status = "No selection made in file browser."
            return
        self.station_xml_path = str(selected)

    def _append_miniseed_from_selection(self, _event=None):
        if not self._browser.value:
            self.status = "No selections made in file browser."
            return

        existing = [item for item in self._split_paths(self.miniseed_files) if item]
        for item in self._browser.value:
            if item not in existing:
                existing.append(item)
        self.miniseed_files = "\n".join(existing)

    def _update_input_type_for_client(self, *_events):
        options = self._INPUT_TYPE_BY_CLIENT[self.client_type]
        self.param.input_type.objects = options
        self.input_type = options[0]

    @staticmethod
    def _split_paths(raw: str) -> list[str]:
        values = []
        for chunk in raw.replace("\n", ",").split(","):
            candidate = chunk.strip()
            if candidate:
                values.append(candidate)
        return values

    def _parse_sample_rates(self) -> list[float]:
        text = self.sample_rates.strip()
        if not text:
            return []
        return [float(item.strip()) for item in text.split(",") if item.strip()]

    def _safe_path(self, value: str) -> str | None:
        cleaned = value.strip()
        return cleaned if cleaned else None

    def _common_kwargs(self) -> dict[str, Any]:
        compression = self.h5_compression
        if compression == "none":
            compression = None

        return {
            "mth5_version": self.mth5_version,
            "mth5_file_mode": self.mth5_file_mode,
            "interact": self.interact,
            "h5_compression": compression,
            "h5_compression_opts": self.h5_compression_opts,
            "h5_shuffle": self.h5_shuffle,
            "h5_fletcher32": self.h5_fletcher32,
            "h5_data_level": self.h5_data_level,
        }

    def _extra_kwargs(self) -> dict[str, Any]:
        text = self.extra_kwargs_json.strip()
        if not text:
            return {}
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("extra_kwargs_json must be a JSON object.")
        return parsed

    def _load_request_csv(self) -> pd.DataFrame:
        request_path = self._safe_path(self.request_csv)
        if request_path is None:
            raise ValueError("request_csv is required for this client type.")
        return pd.read_csv(request_path)

    def _run_create(self):
        try:
            common = self._common_kwargs()
            extras = self._extra_kwargs()

            if self.client_type == "FDSN Client":
                request_df = self._load_request_csv()
                result = MakeMTH5.from_fdsn_client(
                    request_df,
                    client=self.fdsn_client,
                    save_path=self._safe_path(self.save_path),
                    mth5_filename=self._safe_path(self.mth5_filename),
                    **common,
                    **extras,
                )

            elif self.client_type == "FDSN StationXML + miniSEED":
                station_xml = self._safe_path(self.station_xml_path)
                miniseed = self._split_paths(self.miniseed_files)
                if station_xml is None:
                    raise ValueError("station_xml_path is required.")
                if not miniseed:
                    raise ValueError("At least one miniSEED file is required.")
                result = MakeMTH5.from_fdsn_miniseed_and_stationxml(
                    station_xml,
                    miniseed,
                    save_path=self._safe_path(self.save_path),
                    mth5_filename=self._safe_path(self.mth5_filename),
                    **common,
                    **extras,
                )

            elif self.client_type == "USGS Geomag":
                request_df = self._load_request_csv()
                result = MakeMTH5.from_usgs_geomag(
                    request_df,
                    save_path=self._safe_path(self.save_path),
                    mth5_filename=self._safe_path(self.mth5_filename),
                    **common,
                    **extras,
                )

            elif self.client_type == "INTERMAG":
                request_df = self._load_request_csv()
                result = MakeMTH5.from_intermag(
                    request_df,
                    save_path=self._safe_path(self.save_path),
                    mth5_filename=self._safe_path(self.mth5_filename),
                    **common,
                    **extras,
                )

            elif self.client_type == "Zen":
                sample_rates = self._parse_sample_rates()
                data_path = self._safe_path(self.data_path)
                if data_path is None:
                    raise ValueError("data_path is required.")
                result = MakeMTH5.from_zen(
                    data_path,
                    sample_rates=sample_rates or [4096, 1024, 256],
                    calibration_path=self._safe_path(self.calibration_path),
                    survey_id=self._safe_path(self.survey_id),
                    combine=self.combine,
                    save_path=self._safe_path(self.save_path),
                    mth5_filename=self._safe_path(self.mth5_filename),
                    **common,
                    **extras,
                )

            elif self.client_type == "Phoenix":
                sample_rates = self._parse_sample_rates()
                data_path = self._safe_path(self.data_path)
                if data_path is None:
                    raise ValueError("data_path is required.")
                result = MakeMTH5.from_phoenix(
                    data_path,
                    mth5_filename=self._safe_path(self.mth5_filename),
                    save_path=self._safe_path(self.save_path),
                    sample_rates=sample_rates or [150, 24000],
                    receiver_calibration_dict=self._safe_path(
                        self.receiver_calibration_path
                    ),
                    sensor_calibration_dict=self._safe_path(self.sensor_calibration_path),
                    **common,
                    **extras,
                )

            elif self.client_type == "LEMI":
                sample_rates = self._parse_sample_rates()
                data_path = self._safe_path(self.data_path)
                survey_id = self._safe_path(self.survey_id)
                station_id = self._safe_path(self.station_id)
                if data_path is None or survey_id is None or station_id is None:
                    raise ValueError("data_path, survey_id, and station_id are required.")
                result = MakeMTH5.from_lemi(
                    data_path,
                    survey_id,
                    station_id,
                    sample_rates=sample_rates or None,
                    mth5_filename=self._safe_path(self.mth5_filename),
                    save_path=self._safe_path(self.save_path),
                    **common,
                    **extras,
                )

            elif self.client_type == "LEMI-424":
                sample_rates = self._parse_sample_rates()
                data_path = self._safe_path(self.data_path)
                survey_id = self._safe_path(self.survey_id)
                station_id = self._safe_path(self.station_id)
                if data_path is None or survey_id is None or station_id is None:
                    raise ValueError("data_path, survey_id, and station_id are required.")
                result = MakeMTH5.from_lemi424(
                    data_path,
                    survey_id,
                    station_id,
                    sample_rates=sample_rates or None,
                    mth5_filename=self._safe_path(self.mth5_filename),
                    save_path=self._safe_path(self.save_path),
                    **common,
                    **extras,
                )

            elif self.client_type == "LEMI-417":
                data_path = self._safe_path(self.data_path)
                survey_id = self._safe_path(self.survey_id)
                station_id = self._safe_path(self.station_id)
                if data_path is None or survey_id is None or station_id is None:
                    raise ValueError("data_path, survey_id, and station_id are required.")
                result = MakeMTH5.from_lemi417(
                    data_path,
                    survey_id,
                    station_id,
                    mth5_filename=self._safe_path(self.mth5_filename)
                    or "from_lemi417.h5",
                    save_path=self._safe_path(self.save_path) or str(Path.cwd()),
                    **common,
                    **extras,
                )

            elif self.client_type == "Metronix":
                sample_rates = self._parse_sample_rates()
                data_path = self._safe_path(self.data_path)
                if data_path is None:
                    raise ValueError("data_path is required.")
                result = MakeMTH5.from_metronix(
                    data_path,
                    sample_rates=sample_rates or [128],
                    mth5_filename=self._safe_path(self.mth5_filename),
                    save_path=self._safe_path(self.save_path),
                    run_name_zeros=self.run_name_zeros,
                    **common,
                    **extras,
                )

            elif self.client_type == "NIMS":
                sample_rates = self._parse_sample_rates()
                data_path = self._safe_path(self.data_path)
                if data_path is None:
                    raise ValueError("data_path is required.")
                result = MakeMTH5.from_nims(
                    data_path,
                    sample_rates=sample_rates or [4096, 1024, 256],
                    save_path=self._safe_path(self.save_path),
                    calibration_path=self._safe_path(self.calibration_path),
                    survey_id=self._safe_path(self.survey_id),
                    combine=self.combine,
                    mth5_filename=self._safe_path(self.mth5_filename),
                    **common,
                    **extras,
                )

            elif self.client_type == "UoA":
                data_path = self._safe_path(self.data_path)
                survey_id = self._safe_path(self.survey_id)
                station_id = self._safe_path(self.station_id)
                if data_path is None or survey_id is None or station_id is None:
                    raise ValueError("data_path, survey_id, and station_id are required.")
                result = MakeMTH5.from_uoa(
                    data_path,
                    survey_id,
                    station_id,
                    instrument_type=self.instrument_type,
                    run_id=self.run_id,
                    mth5_filename=self._safe_path(self.mth5_filename),
                    save_path=self._safe_path(self.save_path),
                    **common,
                    **extras,
                )

            else:
                raise ValueError(f"Unsupported client_type: {self.client_type}")

            self.status = f"Created MTH5 successfully: {result}"

        except Exception as error:  # pragma: no cover - UI path with broad user inputs
            self.status = f"Error: {error}"

    @pn.depends("client_type")
    def client_help(self):
        return pn.pane.Markdown(
            f"### Input Type\n{self._FILE_TYPE_HELP[self.client_type]}"
        )

    @pn.depends("client_type")
    def client_specific_controls(self):
        controls: list[Any] = []

        if self.client_type in ["FDSN Client", "USGS Geomag", "INTERMAG"]:
            controls.extend(["request_csv"])

        if self.client_type == "FDSN Client":
            controls.append("fdsn_client")

        if self.client_type == "FDSN StationXML + miniSEED":
            controls.extend(["station_xml_path", "miniseed_files"])

        if self.client_type in [
            "Zen",
            "Phoenix",
            "LEMI",
            "LEMI-424",
            "LEMI-417",
            "Metronix",
            "NIMS",
            "UoA",
        ]:
            controls.extend(["data_path", "sample_rates"])

        if self.client_type in ["Zen", "NIMS"]:
            controls.extend(["calibration_path", "combine"])

        if self.client_type in ["LEMI", "LEMI-424", "LEMI-417", "UoA", "NIMS", "Zen"]:
            controls.extend(["survey_id", "station_id"])

        if self.client_type == "Phoenix":
            controls.extend(["receiver_calibration_path", "sensor_calibration_path"])

        if self.client_type == "Metronix":
            controls.append("run_name_zeros")

        if self.client_type == "UoA":
            controls.extend(["instrument_type", "run_id"])

        return pn.Param(
            self,
            parameters=controls,
            show_name=False,
            widgets={
                "miniseed_files": pn.widgets.TextAreaInput,
                "extra_kwargs_json": pn.widgets.TextAreaInput,
            },
        )

    def view(self):
        browser_tools = pn.Column(
            "### File Browser",
            self._browser,
            pn.Row(
                self._set_data_button,
                self._set_save_button,
                self._set_request_button,
            ),
            pn.Row(self._set_stationxml_button, self._set_miniseed_button),
            sizing_mode="stretch_width",
        )

        common_controls = pn.Param(
            self,
            parameters=[
                "client_type",
                "input_type",
                "save_path",
                "mth5_filename",
                "mth5_version",
                "mth5_file_mode",
                "interact",
                "h5_compression",
                "h5_compression_opts",
                "h5_shuffle",
                "h5_fletcher32",
                "h5_data_level",
                "extra_kwargs_json",
                "run",
            ],
            show_name=False,
            widgets={"extra_kwargs_json": pn.widgets.TextAreaInput},
        )

        return pn.Column(
            pn.pane.Markdown("## MakeMTH5 Builder"),
            self.client_help,
            pn.Row(
                pn.Column(common_controls, self.client_specific_controls),
                browser_tools,
            ),
            pn.pane.Markdown("### Status"),
            pn.bind(lambda message: pn.pane.Str(message), self.param.status),
        )


def build_app() -> pn.Column:
    """Return the panel application layout."""
    app = MakeMTH5PanelApp()
    return app.view()


panel_app = build_app()
panel_app.servable()
