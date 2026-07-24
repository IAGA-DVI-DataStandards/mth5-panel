from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import param
import panel as pn

try:
    from .make_mth5 import MakeMTH5PanelApp
except ImportError:  # pragma: no cover - supports running file directly via panel serve
    module_dir = Path(__file__).resolve().parent
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))
    from make_mth5 import MakeMTH5PanelApp

if TYPE_CHECKING:
    from mth5_panel.mth5_viewer import MTH5Viewer

pn.extension("tabulator", sizing_mode="stretch_width")


class MTH5MasterPanelApp(param.Parameterized):
    """Unified panel app for creating and viewing MTH5 files."""

    status = param.String(default="Ready")

    @staticmethod
    def _resolve_logo_path() -> Path | None:
        repo_root = Path(__file__).resolve().parents[2]
        package_root = Path(__file__).resolve().parent
        candidates = [
            package_root / "assets" / "mth5_logo.png",
            repo_root / "mth5" / "docs" / "images" / "mth5_logo.png",
            repo_root / "mth5" / "docs" / "source" / "images" / "mth5_logo.png",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def __init__(
        self,
        make_app: MakeMTH5PanelApp | None = None,
        viewer_app: "MTH5Viewer | None" = None,
        **params: Any,
    ):
        super().__init__(**params)

        self.make_app = make_app or MakeMTH5PanelApp()
        if viewer_app is None:
            try:
                from .mth5_viewer import MTH5Viewer
            except ImportError:  # pragma: no cover - supports running file directly
                from mth5_viewer import MTH5Viewer

            viewer_app = MTH5Viewer(use_template=False)
        self.viewer_app = viewer_app

        self._latest_created_path = ""

        self._handoff_button = pn.widgets.Button(
            name="Open Created File In Viewer",
            button_type="primary",
            disabled=True,
            sizing_mode="stretch_width",
        )
        self._handoff_button.on_click(self._open_created_in_viewer)

        self._created_path_display = pn.widgets.TextInput(
            name="Last Created MTH5",
            value="",
            disabled=True,
            sizing_mode="stretch_width",
        )

        self._status_pane = pn.pane.Alert(
            self.status,
            alert_type="light",
            sizing_mode="stretch_width",
        )

        self.make_app.param.watch(self._on_created_path_changed, "created_mth5_path")
        self.param.watch(self._on_status_changed, "status")

        self._create_tab = pn.Column(
            self.make_app.view(),
            pn.Row(
                self._created_path_display,
                self._handoff_button,
                sizing_mode="stretch_width",
            ),
            self._status_pane,
            sizing_mode="stretch_width",
        )

        self._viewer_tab = pn.Column(
            pn.pane.Markdown("### View MTH5"),
            self.viewer_app.view(),
            sizing_mode="stretch_both",
        )

        self.tabs = pn.Tabs(
            ("Create MTH5", self._create_tab),
            ("View MTH5", self._viewer_tab),
            dynamic=False,
        )

        self.template = pn.template.MaterialTemplate(
            title="",
            header_background="#0b3d5c",
        )

        logo_path = self._resolve_logo_path()
        if logo_path is not None:
            logo = pn.pane.PNG(str(logo_path), height=40, sizing_mode="fixed")
            self.template.header.append(
                pn.Row(
                    logo,
                    pn.Spacer(width=10),
                    # pn.pane.Markdown("### MTH5 Panel"),
                    sizing_mode="stretch_width",
                )
            )

        self.template.main[:] = [self.tabs]

    def _on_created_path_changed(self, event):
        self._latest_created_path = str(event.new).strip()
        self._created_path_display.value = self._latest_created_path
        self._handoff_button.disabled = not bool(self._latest_created_path)
        if self._latest_created_path:
            self.status = "Created file is ready for handoff to the viewer."

    def _on_status_changed(self, event):
        self._status_pane.object = event.new
        if str(event.new).lower().startswith("error"):
            self._status_pane.alert_type = "danger"
        else:
            self._status_pane.alert_type = "light"

    def _open_created_in_viewer(self, _event=None):
        if not self._latest_created_path:
            self.status = "Error: no created MTH5 path is available yet."
            return

        candidate = Path(self._latest_created_path).expanduser()
        if not candidate.exists():
            self.status = f"Error: created MTH5 path does not exist: {candidate}"
            return

        self.viewer_app.load_files([str(candidate)])
        self.viewer_app.tabs.active = 1
        self.tabs.active = 1
        self.status = f"Loaded {candidate} in viewer."

    def view(self):
        return self.template


def build_app():
    app = MTH5MasterPanelApp()
    return app.view()


try:
    panel_app = build_app()
except ImportError as error:  # pragma: no cover - optional-dependency runtime guard
    panel_app = pn.pane.Alert(
        f"Unable to initialize master panel app: {error}",
        alert_type="danger",
        sizing_mode="stretch_width",
    )

panel_app.servable()
