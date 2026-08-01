from __future__ import annotations

import argparse
from typing import Sequence

import panel as pn

from .mth5_panel_app import build_app


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mth5-panel-app",
        description="Launch the MTH5 Panel application.",
    )
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--address", default=None)
    parser.add_argument("--title", default="MTH5 Panel")
    parser.add_argument("--threaded", action="store_true")
    parser.add_argument("--admin", action="store_true")
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        pn.serve(
            {args.title: build_app()},
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
