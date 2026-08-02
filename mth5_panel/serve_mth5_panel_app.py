from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import panel as pn


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mth5-panel-app",
        description="Launch the MTH5 Panel application.",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--address", default=None)
    parser.add_argument("--title", default="MTH5 Panel")
    parser.add_argument(
        "--route",
        default="/",
        help="URL route to serve the app on (for example: / or /mth5-panel).",
    )
    parser.add_argument("--threaded", action="store_true")
    parser.add_argument("--admin", action="store_true")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Build the application and exit without starting the Panel server.",
    )
    return parser


def _ensure_obspy_release_version() -> None:
    """Create an ObsPy release-version file for frozen-runtime imports."""

    release_dir = Path.cwd() / "obspy"
    release_file = release_dir / "RELEASE-VERSION"

    try:
        release_dir.mkdir(parents=True, exist_ok=True)
        if not release_file.exists():
            release_file.write_text("0.0.0+archive\n", encoding="ascii")
    except OSError:
        # Non-fatal: if this fails, normal import error handling will still apply.
        pass


def _build_panel_app():
    _ensure_obspy_release_version()
    try:
        from mth5_panel.mth5_panel_app import build_app
    except ImportError:  # pragma: no cover - supports direct script execution
        from .mth5_panel_app import build_app

    return build_app()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.self_check:
        _build_panel_app()
        return 0

    try:
        app = _build_panel_app()
        pn.serve(
            {args.route: app},
            port=args.port,
            address=args.address,
            show=args.show,
            title=args.title,
            threaded=args.threaded,
            admin=args.admin,
        )
    except KeyboardInterrupt:  # pragma: no cover - interactive shutdown path
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
