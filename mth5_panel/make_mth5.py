# -*- coding: utf-8 -*-
"""Panel application for building MTH5 files with MakeMTH5.

Run with:

    panel serve mth5/clients/make_mth5_panel.py --show
"""

from __future__ import annotations

import io
import json
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import param
from mth5 import logger as mth5_logger
from mth5.clients.make_mth5 import MakeMTH5

try:
    import panel as pn
except (
    ImportError
) as error:  # pragma: no cover - only raised in missing optional dep envs
    raise ImportError(
        "panel and param are required for the MakeMTH5 panel app. "
        "Install with `pip install panel param`."
    ) from error


pn.extension("tabulator", sizing_mode="stretch_width")


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

    mth5_version = param.Selector(
        default="0.2.0",
        objects=["0.2.0", "0.1.0"],
        doc="MTH5 version to use for output file. 0.1.0 is the legacy version, 0.2.0 is the current version.",
    )
    mth5_file_mode = param.Selector(
        default="w",
        objects=["w", "a", "r+"],
        doc="File mode for MTH5 output file. 'w' for write (overwrite), 'a' for append, 'r+' for read/write.",
    )
    # interact = param.Boolean(default=False)

    h5_compression = param.Selector(
        default="gzip", objects=["gzip", "lzf", "szip", "none"]
    )
    h5_compression_opts = param.Integer(default=4, bounds=(0, 9))
    h5_shuffle = param.Boolean(default=True)
    h5_fletcher32 = param.Boolean(default=True)
    h5_data_level = param.Selector(default=1, objects=[0, 1, 2, 3, 4, 5])

    request_csv = param.String(
        default="", doc="Path to request CSV for FDSN/Geomag/INTERMAG."
    )
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
        doc='Extra kwargs as JSON. Example: {"sample_rate": 10.0, "dipole_length_ex": 50.0}',
    )

    status = param.String(default="Ready")
    created_mth5_path = param.String(
        default="",
        doc="Absolute or relative path to last successfully created MTH5 file.",
    )

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
        "FDSN Client": (
            "Input a request CSV with columns: network, station, location, "
            "channel, start, end. Be sure to look up the correct "
            "network/station/location/channel codes for your data source. "
            "see https://www.fdsn.org/webservices/FDSN-WS-Specifications-1.1.pdf "
            "for details, and https://service.iris.edu/fdsnws/availability/1/ "
            "for a list of available networks and stations."
        ),
        "FDSN StationXML + miniSEED": (
            "Identify the StationXML file and one or more " "miniSEED files."
        ),
        "USGS Geomag": (
            "Input a request CSV with geomag columns: observatory, type, "
            "elements, sampling_period, start, end. Times should be in UTC. See "
            "https://www.usgs.gov/observatories/geomag-network/geomag-data-access"
        ),
        "INTERMAG": (
            "Input a request CSV with geomag columns: observatory, type, "
            "elements, sampling_period, start, end. Times should be in UTC. See "
            "https://www.intermagnet.org for more details."
        ),
        "Zen": (
            "Provide the directory containing Z3D files and optional calibration "
            "file, usually amtant.cal."
        ),
        "Phoenix": (
            "Provide the directory containing Phoenix station folders/files and "
            "optional rxcal/scal calibration directories. "
            "Be sure the rxcal and scal have been converted to JSON format using "
            "the `phoenix_calibration_to_json` utility."
        ),
        "LEMI": ("Provide the directory containing LEMI files (.txt or .B423)."),
        "LEMI-424": ("Provide the directory containing LEMI-424 files (.txt)."),
        "LEMI-417": ("Provide the directory containing LEMI-417 files."),
        "Metronix": (
            "Survey or station directory containing Metronix ATSS + JSON structure."
        ),
        "NIMS": (
            "Provide the directory containing NIMS .BIN files and optional calibration file."
        ),
        "UoA": ("Provide the directory or file for UoA PR6-24 or Orange Box data."),
    }

    _BROWSER_ACTIONS_BY_CLIENT = {
        "FDSN Client": ["set_save_path", "set_request_csv"],
        "FDSN StationXML + miniSEED": [
            "set_save_path",
            "set_station_xml",
            "add_miniseed",
        ],
        "USGS Geomag": ["set_save_path", "set_request_csv"],
        "INTERMAG": ["set_save_path", "set_request_csv"],
        "Zen": ["set_save_path", "set_data_path", "set_calibration_path"],
        "Phoenix": [
            "set_save_path",
            "set_data_path",
            "set_receiver_calibration_path",
            "set_sensor_calibration_path",
        ],
        "LEMI": ["set_save_path", "set_data_path"],
        "LEMI-424": ["set_save_path", "set_data_path"],
        "LEMI-417": ["set_save_path", "set_data_path"],
        "Metronix": ["set_save_path", "set_data_path"],
        "NIMS": ["set_save_path", "set_data_path", "set_calibration_path"],
        "UoA": ["set_save_path", "set_data_path"],
    }

    _BROWSER_ACTION_LABELS = {
        "set_save_path": "Use Selected as Save Path",
        "set_data_path": "Use Selected as Data Path",
        "set_request_csv": "Use Selected as Request CSV",
        "set_station_xml": "Use Selected as StationXML",
        "add_miniseed": "Add Selected to miniSEED List",
        "set_calibration_path": "Use Selected as Calibration Path",
        "set_receiver_calibration_path": "Use Selected as Rx Cal Path",
        "set_sensor_calibration_path": "Use Selected as Sensor Cal Path",
    }

    _BROWSER_PARAM_TO_ACTION = {
        "save_path": "set_save_path",
        "data_path": "set_data_path",
        "request_csv": "set_request_csv",
        "station_xml_path": "set_station_xml",
        "calibration_path": "set_calibration_path",
        "receiver_calibration_path": "set_receiver_calibration_path",
        "sensor_calibration_path": "set_sensor_calibration_path",
        "miniseed_files": "add_miniseed",
    }

    def __init__(self, **params: Any):
        super().__init__(**params)

        self._browser = pn.widgets.FileSelector(
            name="Select Files or Directories",
            directory="~",
            file_pattern="*",
            sizing_mode="stretch_width",
        )
        self._browser_actions_panel = pn.GridBox(
            ncols=2,
            sizing_mode="stretch_width",
        )
        self._browser_action_buttons: dict[str, pn.widgets.Button] = {}
        self._status_display = pn.pane.HTML(
            self._format_status_html(self.status),
            height=260,
            sizing_mode="stretch_width",
            styles={
                "overflow-y": "auto",
                "white-space": "pre-wrap",
                "font-family": "Consolas, 'Courier New', monospace",
                "font-size": "0.9rem",
                "padding": "0.5rem",
                "border": "1px solid #d0d7de",
                "border-radius": "0.375rem",
                "background": "#f6f8fa",
            },
        )

        self._browser.param.watch(self._on_browser_selection_changed, "value")

        self._h5_compression_menu = pn.widgets.MenuButton(
            name=f"Comp: {self.h5_compression}",
            items=[(item, item) for item in self.param.h5_compression.objects],
            button_type="light",
            width=170,
        )
        self._h5_compression_opts_menu = pn.widgets.MenuButton(
            name=f"Opts: {self.h5_compression_opts}",
            items=[(str(item), str(item)) for item in range(10)],
            button_type="light",
            width=170,
        )
        self._h5_shuffle_menu = pn.widgets.MenuButton(
            name=f"Shuffle: {self.h5_shuffle}",
            items=[("True", "True"), ("False", "False")],
            button_type="light",
            width=170,
        )
        self._h5_fletcher32_menu = pn.widgets.MenuButton(
            name=f"F32: {self.h5_fletcher32}",
            items=[("True", "True"), ("False", "False")],
            button_type="light",
            width=170,
        )
        self._h5_data_level_menu = pn.widgets.MenuButton(
            name=f"Level: {self.h5_data_level}",
            items=[(str(item), str(item)) for item in self.param.h5_data_level.objects],
            button_type="light",
            width=170,
        )

        self._h5_compression_menu.param.watch(
            self._on_h5_compression_selected, "clicked"
        )
        self._h5_compression_opts_menu.param.watch(
            self._on_h5_compression_opts_selected, "clicked"
        )
        self._h5_shuffle_menu.param.watch(self._on_h5_shuffle_selected, "clicked")
        self._h5_fletcher32_menu.param.watch(self._on_h5_fletcher32_selected, "clicked")
        self._h5_data_level_menu.param.watch(self._on_h5_data_level_selected, "clicked")

        self.param.watch(self._update_input_type_for_client, "client_type")
        self.param.watch(
            self._on_browser_target_param_changed,
            list(self._BROWSER_PARAM_TO_ACTION.keys()),
        )
        self.param.watch(self._on_status_changed, "status")
        self._update_input_type_for_client()

    def _on_browser_selection_changed(self, event):
        if event.new:
            self.status = f"Selected {len(event.new)} item(s) in file browser."

    def _set_browser_action_state(self, action_key: str, success: bool):
        button = self._browser_action_buttons.get(action_key)
        if button is not None:
            button.button_type = "success" if success else "danger"

    @staticmethod
    def _format_status_html(message: str) -> str:
        return escape(message)

    def _on_status_changed(self, event):
        self._status_display.object = self._format_status_html(event.new)

    def _on_browser_target_param_changed(self, event):
        action_key = self._BROWSER_PARAM_TO_ACTION.get(event.name)
        if action_key is None:
            return

        if event.name == "miniseed_files":
            success = len(self._split_paths(str(event.new))) > 0
        else:
            success = bool(str(event.new).strip())

        self._set_browser_action_state(action_key, success=success)

    def _set_path_parameter_from_selection(
        self,
        parameter_name: str,
        action_key: str,
        require_directory: bool = False,
    ):
        selected = self._selected_path()
        if selected is None:
            self.status = "No selection made in file browser."
            self._set_browser_action_state(action_key, success=False)
            return

        if require_directory:
            value = selected if selected.is_dir() else selected.parent
        else:
            value = selected

        setattr(self, parameter_name, str(value))
        self.status = f"Set {parameter_name} to: {value}"
        self._set_browser_action_state(action_key, success=True)

    def _clear_browser_selection(self):
        self._browser.value = []

    def _run_browser_action(self, action_key: str, _event=None):
        try:
            if action_key == "set_save_path":
                self._set_path_parameter_from_selection(
                    "save_path", action_key, require_directory=True
                )
            elif action_key == "set_data_path":
                self._set_path_parameter_from_selection(
                    "data_path", action_key, require_directory=True
                )
            elif action_key == "set_request_csv":
                self._set_path_parameter_from_selection("request_csv", action_key)
            elif action_key == "set_station_xml":
                self._set_path_parameter_from_selection("station_xml_path", action_key)
            elif action_key == "set_calibration_path":
                self._set_path_parameter_from_selection("calibration_path", action_key)
            elif action_key == "set_receiver_calibration_path":
                self._set_path_parameter_from_selection(
                    "receiver_calibration_path", action_key
                )
            elif action_key == "set_sensor_calibration_path":
                self._set_path_parameter_from_selection(
                    "sensor_calibration_path", action_key
                )
            elif action_key == "add_miniseed":
                if not self._browser.value:
                    self.status = "No selections made in file browser."
                    self._set_browser_action_state(action_key, success=False)
                    return

                existing = [
                    item for item in self._split_paths(self.miniseed_files) if item
                ]
                for item in self._browser.value:
                    if item not in existing:
                        existing.append(item)

                self.miniseed_files = "\n".join(existing)
                self.status = f"Added {len(self._browser.value)} miniSEED path(s)."
                self._set_browser_action_state(action_key, success=True)
        finally:
            self._clear_browser_selection()

    def _update_browser_actions(self):
        action_keys = self._BROWSER_ACTIONS_BY_CLIENT.get(self.client_type, [])
        self._browser_action_buttons = {}

        buttons: list[pn.widgets.Button] = []
        for action_key in action_keys:
            button = pn.widgets.Button(
                name=self._BROWSER_ACTION_LABELS[action_key],
                button_type="danger",
                sizing_mode="stretch_width",
            )
            button.on_click(
                lambda event, key=action_key: self._run_browser_action(key, event)
            )
            self._browser_action_buttons[action_key] = button
            buttons.append(button)

        self._browser_actions_panel.objects = buttons

    @staticmethod
    def _to_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _on_h5_compression_selected(self, event):
        value = event.new
        if value in self.param.h5_compression.objects:
            self.h5_compression = value
            self._h5_compression_menu.name = f"Comp: {self.h5_compression}"

    def _on_h5_compression_opts_selected(self, event):
        self.h5_compression_opts = int(event.new)
        self._h5_compression_opts_menu.name = f"Opts: {self.h5_compression_opts}"

    def _on_h5_shuffle_selected(self, event):
        self.h5_shuffle = self._to_bool(event.new)
        self._h5_shuffle_menu.name = f"Shuffle: {self.h5_shuffle}"

    def _on_h5_fletcher32_selected(self, event):
        self.h5_fletcher32 = self._to_bool(event.new)
        self._h5_fletcher32_menu.name = f"F32: {self.h5_fletcher32}"

    def _on_h5_data_level_selected(self, event):
        value = int(event.new)
        if value in self.param.h5_data_level.objects:
            self.h5_data_level = value
            self._h5_data_level_menu.name = f"Level: {self.h5_data_level}"

    def _selected_path(self) -> Path | None:
        if not self._browser.value:
            return None
        return Path(self._browser.value[0])

    def _update_input_type_for_client(self, *_events):
        options = self._INPUT_TYPE_BY_CLIENT[self.client_type]
        self.param.input_type.objects = options
        self.input_type = options[0]
        self._update_browser_actions()

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
            # "interact": self.interact,
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
        log_buffer = io.StringIO()
        stdout_buffer = io.StringIO()
        log_sink_id = None
        try:
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(stdout_buffer))
                stack.enter_context(redirect_stderr(stdout_buffer))
                log_sink_id = mth5_logger.add(
                    log_buffer,
                    level="INFO",
                    colorize=False,
                    format="{time} | {level: <8} | {name}:{function}:{line} | {message}",
                )
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
                        sensor_calibration_dict=self._safe_path(
                            self.sensor_calibration_path
                        ),
                        **common,
                        **extras,
                    )

                elif self.client_type == "LEMI":
                    sample_rates = self._parse_sample_rates()
                    data_path = self._safe_path(self.data_path)
                    survey_id = self._safe_path(self.survey_id)
                    station_id = self._safe_path(self.station_id)
                    if data_path is None or survey_id is None or station_id is None:
                        raise ValueError(
                            "data_path, survey_id, and station_id are required."
                        )
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
                        raise ValueError(
                            "data_path, survey_id, and station_id are required."
                        )
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
                        raise ValueError(
                            "data_path, survey_id, and station_id are required."
                        )
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
                        raise ValueError(
                            "data_path, survey_id, and station_id are required."
                        )
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

            self.created_mth5_path = str(result) if result is not None else ""
            status_lines = [f"Created MTH5 successfully: {result}"]
            log_text = log_buffer.getvalue().strip()
            if log_text:
                status_lines.extend(["", "Logs:", log_text])
            stdout_text = stdout_buffer.getvalue().strip()
            if stdout_text:
                status_lines.extend(["", "Stdout:", stdout_text])
            self.status = "\n".join(status_lines)

        except Exception as error:  # pragma: no cover - UI path with broad user inputs
            self.created_mth5_path = ""
            status_lines = [f"Error: {error}"]
            log_text = log_buffer.getvalue().strip()
            if log_text:
                status_lines.extend(["", "Logs:", log_text])
            stdout_text = stdout_buffer.getvalue().strip()
            if stdout_text:
                status_lines.extend(["", "Stdout:", stdout_text])
            self.status = "\n".join(status_lines)
        finally:
            if log_sink_id is not None:
                mth5_logger.remove(log_sink_id)

    @pn.depends("client_type")
    def client_help(self):
        return pn.pane.Markdown(
            f"### For the Selected Client\n{self._FILE_TYPE_HELP[self.client_type]}"
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
            self._browser,
            self._browser_actions_panel,
            pn.layout.Divider(),
            self._status_display,
            sizing_mode="stretch_width",
        )

        h5_controls_grid = pn.GridBox(
            self._h5_compression_menu,
            self._h5_data_level_menu,
            self._h5_compression_opts_menu,
            self._h5_shuffle_menu,
            self._h5_fletcher32_menu,
            ncols=2,
            sizing_mode="stretch_width",
        )

        common_controls = pn.Param(
            self,
            parameters=[
                "client_type",
                # "input_type",
                "save_path",
                "mth5_filename",
                "mth5_version",
                "mth5_file_mode",
                # "interact",
                "extra_kwargs_json",
            ],
            show_name=False,
            widgets={"extra_kwargs_json": pn.widgets.TextAreaInput},
        )

        run_button = pn.widgets.Button.from_param(
            self.param.run,
            name="Create MTH5",
            button_type="primary",
            sizing_mode="stretch_width",
        )

        return pn.Column(
            pn.pane.Markdown("## MakeMTH5 Builder"),
            "### Instructions",
            (
                "**Step 1**: Select the client type from the dropdown menu. This will determine the required "
                "input files and directories needed to create the MTH5 file.\n\n"
                "**Step 2**: Use the file browser below to select the required files or directories. \n"
                "Use the file browser below to select files or directories depending on "
                "what is required for the selected client, which are displayed as red buttons "
                "below. The selected files/directories will be used to populate the corresponding "
                "fields in the form below.  Once you have selected the appropriate files/directories, "
                "click the red buttons to set the values. The buttons will turn green when the "
                "values are set correctly and the selection will be cleared.  You can also "
                "manually enter the paths in the form below if you prefer.\n\n"
                "**Step 3**: Fill in any additional required fields in the form below. Fields  "
                "under the client specific controls that are required.\n\n"
                "**Step 4**: Click the 'Create MTH5' button to generate the MTH5 file. The status of the "
                "operation will be displayed in the status area below the file browser. If the operation "
                "is successful, the path to the created MTH5 file will be displayed. If there are any "
                "errors, they will be displayed in the status area as well.\n\n"
                "**Step 5**: Once the MTH5 file is created, you can use the 'Handoff' button to open the "
                "MTH5 file in the MTH5 Viewer for further inspection and analysis."
            ),
            self.client_help,
            pn.layout.Divider(),
            pn.Row(
                pn.Column(
                    common_controls,
                    pn.layout.Divider(),
                    pn.pane.Markdown("### H5 Options"),
                    h5_controls_grid,
                    pn.layout.Divider(),
                    pn.pane.Markdown(
                        "### <span style='color:red'>REQUIRED</span>  Client Specific Controls",
                    ),
                    self.client_specific_controls,
                    width=400,
                ),
                browser_tools,
            ),
            pn.layout.Divider(),
            run_button,
        )


def build_app() -> pn.Column:
    """Return the panel application layout."""
    app = MakeMTH5PanelApp()
    return app.view()


if __name__.startswith("bokeh_app") or __name__ == "__main__":
    panel_app = build_app()
    panel_app.servable()
