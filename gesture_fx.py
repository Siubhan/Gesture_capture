"""
Gesture FX: жест перед камерой -> анимация поверх видео.

  Open_Palm    ладонь                     -> конфетти
  Closed_Fist  кулак                      -> взрыв с ударной волной
  Victory      два пальца буквой V        -> фейерверк
  Gun          палец-пистолет (2 вариации) -> выстрел: вспышка, трассер, отдача, звук
  Heart        сердечко двумя руками      -> лицо превращается в жабу

Установка:  pip install -r requirements.txt
Запуск:     python gesture_fx.py [--camera N] [--mute] [--profile]

Клавиши: q / Esc - выход, l - показать/скрыть точки руки.
Модели скачиваются автоматически в папку models/ при первом запуске.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2

from effects import FX
from faceswap import FaceSwap
from gestures import HOLD_FRAMES, Debouncer, classify, is_heart, make_hand
from sound import Sound

# ----------------------------- настройки ------------------------------------

FRAME_W, FRAME_H = 960, 540

MODELS_DIR = Path(__file__).resolve().parent / "models"
_GCS = "https://storage.googleapis.com/mediapipe-models"
GESTURE_MODEL = ("gesture_recognizer.task",
                 f"{_GCS}/gesture_recognizer/gesture_recognizer/float16/latest/"
                 "gesture_recognizer.task")
FACE_MODEL = ("blaze_face_short_range.tflite",
              f"{_GCS}/face_detector/blaze_face_short_range/float16/latest/"
              "blaze_face_short_range.tflite")

# жест -> (эффект, пауза между запусками в секундах, подпись в HUD).
# Подписи только латиницей: шрифты OpenCV не умеют кириллицу.
GESTURES = {
    "Open_Palm": (FX.confetti, 0.9, "Palm -> confetti"),
    "Closed_Fist": (FX.explosion, 1.0, "Fist -> explosion"),
    "Victory": (FX.firework, 1.4, "Victory -> fireworks"),
    "Gun": (FX.gunshot, 0.7, "Gun -> BANG!"),
}
HEART_LABEL = "Heart -> frog face"
IDLE_LABEL = "Palm / Fist / Victory / Gun / Heart"


# ----------------------------- модели и камера ------------------------------

def ensure_model(filename: str, url: str) -> str:
    """Путь к модели; скачивает её, если файла ещё нет."""
    path = MODELS_DIR / filename
    if not path.exists():
        print(f"Скачиваю {filename}...")
        MODELS_DIR.mkdir(exist_ok=True)
        part = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, part)
        part.replace(path)  # файл появляется только после полной загрузки
    return str(path)


def create_models(with_face: bool):
    """MediaPipe импортируется здесь, чтобы остальные модули (и тесты) не тянули его."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    gesture = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model(*GESTURE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ))
    face = None
    if with_face:
        face = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model(*FACE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=0.5,
        ))
    return mp, gesture, face


def open_camera(index=None, max_index=6):
    """Открывает камеру. На Windows использует DirectShow (CAP_DSHOW),
    он стабильнее видит внешние USB-камеры. Если индекс не задан,
    перебирает 0..max_index-1 и берёт первую, которая отдаёт кадры."""
    backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
    candidates = [index] if index is not None else range(max_index)
    for i in candidates:
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            ok, _ = cap.read()
            if ok:
                print(f"Камера найдена: индекс {i}")
                return cap
        cap.release()
    return None


# ----------------------------- профилирование -------------------------------

class Profiler:
    """Среднее время этапов кадра (флаг --profile). Без флага ничего не стоит."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.totals: dict[str, float] = {}
        self.frames = 0
        self._t = 0.0

    def start(self):
        if self.enabled:
            self._t = time.perf_counter()
            self.frames += 1

    def lap(self, stage: str):
        if self.enabled:
            t = time.perf_counter()
            self.totals[stage] = self.totals.get(stage, 0.0) + (t - self._t)
            self._t = t

    def report(self) -> str:
        if not self.enabled or not self.frames:
            return ""
        per_frame = {k: v / self.frames * 1000 for k, v in self.totals.items()}
        total = sum(per_frame.values())
        lines = [f"Профиль за {self.frames} кадров (мс на кадр):"]
        lines += [f"  {k:<10} {v:6.2f}" for k, v in per_frame.items()]
        lines.append(f"  {'итого':<10} {total:6.2f}  ->  {1000 / total:.1f} FPS")
        return "\n".join(lines)


# ----------------------------- main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=None,
                    help="индекс камеры (по умолчанию - автопоиск)")
    ap.add_argument("--mute", action="store_true", help="отключить звук выстрела")
    ap.add_argument("--profile", action="store_true",
                    help="печатать среднее время этапов кадра при выходе")
    args = ap.parse_args()

    swap = FaceSwap.load()
    if swap is None:
        print("assets/pepe.png не найден или без прозрачности: замена лица отключена.")
    mp, recognizer, face_detector = create_models(with_face=swap is not None)

    cap = open_camera(args.camera)
    if cap is None:
        raise SystemExit(
            "Не удалось открыть камеру. Проверьте подключение, доступ к камере "
            "в настройках Windows и что её не занимает другое приложение.")

    fx = FX(None if args.mute else Sound())
    debounce = Debouncer({name: g[1] for name, g in GESTURES.items()})
    prof = Profiler(args.profile)
    heart_frames = 0
    show_points = True
    start = prev = time.time()
    last_ts = -1

    while True:
        prof.start()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # зеркало
        h, w = frame.shape[:2]
        prof.lap("capture")

        now = time.time()
        dt = min(now - prev, 0.05)
        prev = now
        ts = max(int((now - start) * 1000), last_ts + 1)  # MediaPipe требует растущие метки
        last_ts = ts

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = recognizer.recognize_for_video(mp_img, ts)
        hands = result.hand_landmarks
        world = getattr(result, "hand_world_landmarks", None) or []
        prof.lap("hands")

        # сердечко (две руки) подавляет обычные жесты, чтобы они не срабатывали заодно
        heart = len(hands) == 2 and is_heart(hands[0], hands[1], w, h)
        heart_frames = heart_frames + 1 if heart else 0
        if heart_frames >= HOLD_FRAMES and swap:
            swap.trigger(now)

        label = HEART_LABEL if heart_frames >= HOLD_FRAMES and swap else None
        for idx, lms in enumerate(hands):
            cats = result.gestures[idx] if idx < len(result.gestures) else []
            name = classify(cats, world[idx] if idx < len(world) else lms)
            fire = debounce.update(idx, name, now)
            if name in GESTURES and not heart_frames:
                label = GESTURES[name][2]
                if fire:
                    GESTURES[name][0](fx, make_hand(lms, w, h))
        debounce.prune(len(hands))
        prof.lap("logic")

        if swap:
            swap.update(now, dt)
            if swap.visible:  # детектор лиц запускаем только пока жаба на экране
                faces = swap.apply(
                    frame, face_detector.detect_for_video(mp_img, ts).detections, w, h)
                if swap.puff_pending and faces:
                    swap.puff_pending = False
                    for f in faces:
                        fx.puff(f.cx, f.cy, f.box_w / 2)
        prof.lap("faceswap")

        if show_points:
            for lms in hands:
                for lm in lms:
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (0, 255, 0), -1)
        fx.update(dt)
        frame = fx.render(frame)
        prof.lap("effects")

        cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.putText(frame, label or IDLE_LABEL, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Gesture FX", frame)
        key = cv2.waitKey(1) & 0xFF
        prof.lap("display")
        if key in (ord("q"), 27):
            break
        if key == ord("l"):
            show_points = not show_points

    cap.release()
    recognizer.close()
    if face_detector:
        face_detector.close()
    cv2.destroyAllWindows()
    if prof.enabled:
        print(prof.report())


if __name__ == "__main__":
    main()
