from __future__ import annotations

from mth5_panel import serve_mth5_panel_app as serve


def test_main_serves_panel_app(monkeypatch):
    calls: dict[str, object] = {}

    sentinel_app = object()

    def _fake_build_app():
        return sentinel_app

    def _fake_serve(panels, **kwargs):
        calls["panels"] = panels
        calls["kwargs"] = kwargs

        return None

    monkeypatch.setattr(serve, "build_app", _fake_build_app)
    monkeypatch.setattr(serve.pn, "serve", _fake_serve)

    exit_code = serve.main(["--port", "5007", "--title", "Custom MTH5 Panel"])

    assert exit_code == 0
    assert calls["panels"] == {"Custom MTH5 Panel": sentinel_app}
    assert calls["kwargs"]["port"] == 5007
    assert calls["kwargs"]["show"] is True
    assert calls["kwargs"]["title"] == "Custom MTH5 Panel"
    assert calls["kwargs"]["threaded"] is False
    assert calls["kwargs"]["admin"] is False
