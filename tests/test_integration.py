"""Сквозной тест main(): подменяем камеру, MediaPipe и окно OpenCV заглушками
и проверяем, что жесты доходят до эффектов, а сердечко - до замены лица."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from helpers import Pt, heart_pair

import gesture_fx as app

DATA = Path(__file__).parent / "data" / "gun_variant2_landmarks.json"
W, H = 960, 540


class FakeCap:
    def __init__(self, frames):
        self.left = frames
        self.released = False

    def read(self):
        if self.left <= 0:
            return False, None
        self.left -= 1
        return True, np.full((H, W, 3), 100, np.uint8)

    def release(self):
        self.released = True


class FakeRecognizer:
    def __init__(self, scripted):
        self.scripted, self.calls, self.closed = scripted, 0, False

    def recognize_for_video(self, img, ts):
        self.calls += 1
        return self.scripted(self.calls)

    def close(self):
        self.closed = True


class FakeFace:
    def __init__(self):
        self.calls, self.closed = 0, False

    def detect_for_video(self, img, ts):
        self.calls += 1
        kp = [SimpleNamespace(x=420 / W, y=240 / H), SimpleNamespace(x=520 / W, y=240 / H)]
        det = SimpleNamespace(keypoints=kp, bounding_box=SimpleNamespace(width=230))
        return SimpleNamespace(detections=[det])

    def close(self):
        self.closed = True


def empty():
    return SimpleNamespace(hand_landmarks=[], gestures=[], hand_world_landmarks=[])


@pytest.fixture
def harness(monkeypatch):
    """Подменяет всё, что требует железа, и возвращает управляющий объект."""
    h = SimpleNamespace(frames=12, scripted=lambda n: empty(), face=FakeFace(), recognizer=None,
                        cap=None, shown=0)

    def fake_open(index=None, max_index=6):
        h.cap = FakeCap(h.frames)
        return h.cap

    def fake_models(with_face):
        mp = SimpleNamespace(Image=lambda image_format, data: object(),
                             ImageFormat=SimpleNamespace(SRGB=0))
        h.recognizer = FakeRecognizer(h.scripted)
        return mp, h.recognizer, h.face if with_face else None

    def fake_imshow(name, frame):
        h.shown += 1
        assert frame.shape == (H, W, 3)

    monkeypatch.setattr(app, "open_camera", fake_open)
    monkeypatch.setattr(app, "create_models", fake_models)
    monkeypatch.setattr(cv2, "imshow", fake_imshow)
    monkeypatch.setattr(cv2, "waitKey", lambda ms: -1)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(app.sys, "argv", ["gesture_fx.py", "--mute"])
    return h


def victory_category():
    return [[SimpleNamespace(category_name="Victory", score=0.95)]]


def gun_result(_):
    data = json.loads(DATA.read_text())
    return SimpleNamespace(
        hand_landmarks=[[Pt(*p) for p in data["norm"]]],
        hand_world_landmarks=[[Pt(*p) for p in data["world"]]],
        gestures=victory_category())  # модель думает «Victory», собственное правило должно перебить


def test_real_gun_variant2_fires_gun_not_fireworks(harness, monkeypatch):
    harness.scripted = gun_result
    shots, fireworks = [], []
    monkeypatch.setitem(app.GESTURES, "Gun", (lambda fx, hand: shots.append(hand), 0.7, "Gun"))
    monkeypatch.setitem(app.GESTURES, "Victory", (lambda fx, hand: fireworks.append(hand), 1.4, "V"))
    app.main()
    assert len(shots) == 1, "выстрел один: дальше работает кулдаун"
    assert not fireworks
    assert shots[0].frame_size == (W, H) and shots[0].aim is not None
    assert harness.shown == harness.frames
    assert harness.cap.released and harness.recognizer.closed and harness.face.closed


def heart_result(_):
    a, b = heart_pair()
    return SimpleNamespace(hand_landmarks=[a, b], hand_world_landmarks=[],
                           gestures=[[SimpleNamespace(category_name="Open_Palm", score=0.9)]] * 2)


def test_heart_swaps_face_and_suppresses_other_gestures(harness, monkeypatch):
    harness.scripted = heart_result
    palms, puffs = [], []
    monkeypatch.setitem(app.GESTURES, "Open_Palm", (lambda fx, hand: palms.append(hand), 0.9, "P"))
    monkeypatch.setattr(app.FX, "puff", lambda self, x, y, r: puffs.append((x, y, r)))
    app.main()
    assert harness.face.calls > 0, "детектор лиц должен запускаться при сердечке"
    assert len(puffs) == 1 and puffs[0][0] == pytest.approx(470, abs=1)
    # первые HOLD_FRAMES-1 кадров ладонь ещё может сработать, но после сердечка - нет
    assert len(palms) <= 1


def test_no_hands_means_no_face_detection_and_no_effects(harness):
    app.main()
    assert harness.face.calls == 0          # детектор лиц спит, пока жабы нет
    assert harness.shown == harness.frames


def test_profile_flag_prints_stage_timings(harness, monkeypatch, capsys):
    monkeypatch.setattr(app.sys, "argv", ["gesture_fx.py", "--mute", "--profile"])
    app.main()
    out = capsys.readouterr().out
    for stage in ("capture", "hands", "logic", "faceswap", "effects", "display", "FPS"):
        assert stage in out


def test_missing_camera_exits_with_message(harness, monkeypatch):
    monkeypatch.setattr(app, "open_camera", lambda index=None, max_index=6: None)
    with pytest.raises(SystemExit) as e:
        app.main()
    assert "камеру" in str(e.value)


def test_missing_sprite_disables_face_swap_but_app_runs(harness, monkeypatch, capsys):
    monkeypatch.setattr(app.FaceSwap, "load", classmethod(lambda cls, *a, **k: None))
    harness.scripted = heart_result
    app.main()
    assert harness.face.calls == 0
    assert "замена лица отключена" in capsys.readouterr().out
