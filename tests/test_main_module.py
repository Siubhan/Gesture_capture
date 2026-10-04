import subprocess
import sys
import time
from pathlib import Path

import gesture_fx as app


def test_importing_app_does_not_load_mediapipe():
    """Лёгкий импорт: без камеры, моделей и сети (MediaPipe подгружается в main)."""
    code = "import gesture_fx, sys; sys.exit(1 if 'mediapipe' in sys.modules else 0)"
    root = Path(__file__).resolve().parent.parent
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0


def test_gesture_table_is_consistent_and_labels_are_ascii():
    """Шрифты OpenCV не умеют кириллицу: она превращается в '???'."""
    for name, (effect, cooldown, label) in app.GESTURES.items():
        assert callable(effect) and cooldown > 0
        assert label.isascii(), f"{name}: кириллица в HUD-подписи"
    assert app.HEART_LABEL.isascii() and app.IDLE_LABEL.isascii()


def test_profiler_disabled_is_silent_and_enabled_reports():
    off = app.Profiler(False)
    off.start(); off.lap("x")
    assert off.report() == ""
    on = app.Profiler(True)
    for _ in range(3):
        on.start(); time.sleep(0.002); on.lap("hands")
    report = on.report()
    assert "hands" in report and "FPS" in report and "3 кадров" in report


def test_ensure_model_downloads_atomically(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MODELS_DIR", tmp_path / "models")
    calls = []

    def fake_retrieve(url, dest):
        calls.append(url)
        dest.write_bytes(b"model")

    monkeypatch.setattr(app.urllib.request, "urlretrieve", fake_retrieve)
    path = app.ensure_model("m.task", "http://x/m.task")
    assert Path(path).read_bytes() == b"model" and len(calls) == 1
    assert not list((tmp_path / "models").glob("*.part"))
    app.ensure_model("m.task", "http://x/m.task")
    assert len(calls) == 1        # второй раз не скачивает
