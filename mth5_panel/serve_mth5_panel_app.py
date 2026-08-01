from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _build_command(extra_args: Sequence[str] | None = None) -> list[str]:
    app_path = Path(__file__).resolve().with_name("mth5_panel_app.py")
    command = [sys.executable, "-m", "panel", "serve", str(app_path), "--show"]
    if extra_args:
        command.extend(extra_args)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    command = _build_command(sys.argv[1:] if argv is None else argv)
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
