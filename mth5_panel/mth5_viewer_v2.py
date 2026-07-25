from __future__ import annotations

from .mth5_viewer import MTH5Viewer as MTH5ViewerV2


def build_app():
    return MTH5ViewerV2(plot_width=750, plot_height=220).view()


if __name__.startswith("bokeh_app") or __name__ == "__main__":
    panel_app = build_app()
    panel_app.servable()
