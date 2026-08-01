from __future__ import annotations

import sys
from pathlib import Path

from mth5_panel import serve_mth5_panel_app as serve


def test_main_builds_panel_serve_command(monkeypatch):
    calls: dict[str, object] = {}

    def _fake_run(command, check=False):
        calls["command"] = command
        calls["check"] = check

        class _Result:
            returncode = 0

        return _Result()

    monkeypatch.setattr(serve.subprocess, "run", _fake_run)

    exit_code = serve.main(["--port", "5007"])

    expected_app = str(Path(serve.__file__).resolve().with_name("mth5_panel_app.py"))
    assert exit_code == 0
    assert calls["check"] is False
    assert calls["command"] == [
        sys.executable,
        "-m",
        "panel",
        "serve",
        expected_app,
        "--show",
        "--port",
        "5007",
    ]
