"""Замена лица картинкой (жаба Pepe) по жесту «сердечко».

Положение, размер и наклон картинки берутся из ключевых точек детектора лиц
MediaPipe (глаза), так что жаба поворачивается вместе с головой.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SPRITE_PATH = Path(__file__).resolve().parent / "assets" / "pepe.png"
# Центры зрачков на картинке assets/pepe.png (x, y в пикселях картинки).
# Если подставите свой спрайт, поменяйте эти координаты.
SPRITE_EYES = ((249.9, 146.5), (422.5, 146.0))

FADE_IN = 0.18       # секунд на появление
FADE_OUT = 0.45      # секунд на исчезновение
LINGER = 1.5         # сколько жаба остаётся после того, как сердечко убрали
SMOOTH = 0.55        # вес нового кадра при сглаживании (1 = без сглаживания)
MISS_FRAMES = 8      # столько кадров держим жабу, если детектор потерял лицо
FROG_EYE_K = 1.0     # масштаб по расстоянию между глазами
FROG_BOX_K = 1.25    # масштаб по ширине рамки лица
ALPHA_VISIBLE = 40   # порог прозрачности, чтобы считать пиксель «головой»


@dataclass(frozen=True)
class FaceBox:
    cx: float      # середина между глазами
    cy: float
    eye: float     # расстояние между глазами, px
    box_w: float   # ширина рамки лица, px
    angle: float   # наклон линии глаз, рад


def face_from_detection(det, w: int, h: int) -> FaceBox | None:
    """FaceBox из детекции MediaPipe. Ключевые точки 0 и 1 - глаза."""
    kp = det.keypoints
    if len(kp) < 2:
        return None
    a = (kp[0].x * w, kp[0].y * h)
    b = (kp[1].x * w, kp[1].y * h)
    if a[0] > b[0]:
        a, b = b, a  # слева направо в кадре
    eye = math.dist(a, b)
    if eye < 1:
        return None
    return FaceBox((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, eye,
                   float(det.bounding_box.width), math.atan2(b[1] - a[1], b[0] - a[0]))


def _lerp(old: FaceBox, new: FaceBox, k: float) -> FaceBox:
    def mix(a: float, b: float) -> float:
        return a + (b - a) * k

    return FaceBox(mix(old.cx, new.cx), mix(old.cy, new.cy), mix(old.eye, new.eye),
                   mix(old.box_w, new.box_w), mix(old.angle, new.angle))


def blend_premultiplied(roi: np.ndarray, warped: np.ndarray, alpha: float) -> None:
    """roi = warped + roi * (1 - a), где warped - BGRA с premultiplied alpha,
    а alpha - общее затухание. Целочисленные операции OpenCV (быстрее float32)."""
    fg = warped if alpha >= 1.0 else cv2.convertScaleAbs(warped, alpha=alpha)
    inv = cv2.bitwise_not(cv2.extractChannel(fg, 3))          # 255 - a
    cv2.multiply(roi, cv2.merge((inv, inv, inv)), dst=roi, scale=1 / 255)
    cv2.add(roi, cv2.cvtColor(fg, cv2.COLOR_BGRA2BGR), dst=roi)


class FaceSwap:
    def __init__(self, sprite_bgra: np.ndarray, eyes=SPRITE_EYES):
        alpha = sprite_bgra[:, :, 3]
        premult = sprite_bgra.astype(np.float32)
        premult[:, :, :3] *= premult[:, :, 3:4] / 255.0  # premultiplied alpha: без каймы
        self.sprite = premult.astype(np.uint8)
        (x1, y1), (x2, y2) = eyes
        self.eye_mid = ((x1 + x2) / 2, (y1 + y2) / 2)
        self.eye_px = math.hypot(x2 - x1, y2 - y1)
        self.angle0 = math.atan2(y2 - y1, x2 - x1)
        xs = np.where((alpha > ALPHA_VISIBLE).any(axis=0))[0]
        self.head_w = float(xs.max() - xs.min())

        self.alpha = 0.0
        self.hold_until = -math.inf
        self.puff_pending = False
        self._faces: list[FaceBox] = []
        self._miss = 0

    @classmethod
    def load(cls, path=SPRITE_PATH, eyes=SPRITE_EYES) -> FaceSwap | None:
        """None, если картинки нет или в ней нет канала прозрачности."""
        try:
            data = np.fromfile(str(path), np.uint8)  # читаем так, чтобы работали не-ASCII пути
        except OSError:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            return None
        return cls(img, eyes)

    # --- состояние ---
    def trigger(self, now: float) -> None:
        """Вызывать каждый кадр, пока держится сердечко."""
        if self.alpha <= 0.0 and now >= self.hold_until:
            self.puff_pending = True
        self.hold_until = now + LINGER

    def update(self, now: float, dt: float) -> None:
        target = 1.0 if now < self.hold_until else 0.0
        if target > self.alpha:
            self.alpha = min(target, self.alpha + dt / FADE_IN)
        else:
            self.alpha = max(target, self.alpha - dt / FADE_OUT)
        if self.alpha <= 0.0:
            self._faces, self._miss = [], 0

    @property
    def visible(self) -> bool:
        return self.alpha > 0.0

    # --- отрисовка ---
    def apply(self, frame: np.ndarray, detections, w: int, h: int) -> list[FaceBox]:
        """Рисует жабу поверх лиц (in-place). Возвращает сглаженные лица."""
        new = sorted((f for f in (face_from_detection(d, w, h) for d in detections) if f),
                     key=lambda f: f.cx)
        if new:
            self._faces = [
                _lerp(self._faces[i], f, SMOOTH) if i < len(self._faces) else f
                for i, f in enumerate(new)]
            self._miss = 0
        elif self._faces and self._miss < MISS_FRAMES:
            self._miss += 1  # детектор моргнул: показываем прошлую позицию
        else:
            self._faces = []
        for face in self._faces:
            self._paste(frame, face)
        return self._faces

    def _paste(self, frame: np.ndarray, f: FaceBox) -> None:
        h, w = frame.shape[:2]
        scale = 0.5 * (f.eye * FROG_EYE_K / self.eye_px + f.box_w * FROG_BOX_K / self.head_w)
        rot = f.angle - self.angle0
        c, s = math.cos(rot) * scale, math.sin(rot) * scale
        ex, ey = self.eye_mid
        m = np.float32([[c, -s, f.cx - (c * ex - s * ey)],
                        [s, c, f.cy - (s * ex + c * ey)]])

        # работаем только с прямоугольником, куда попадает спрайт, а не со всем кадром
        sh, sw = self.sprite.shape[:2]
        corners = np.float32([[0, 0], [sw, 0], [sw, sh], [0, sh]]) @ m[:, :2].T + m[:, 2]
        x0 = max(math.floor(corners[:, 0].min()), 0)
        y0 = max(math.floor(corners[:, 1].min()), 0)
        x1 = min(math.ceil(corners[:, 0].max()), w)
        y1 = min(math.ceil(corners[:, 1].max()), h)
        if x1 <= x0 or y1 <= y0:
            return
        m[0, 2] -= x0
        m[1, 2] -= y0
        warped = cv2.warpAffine(self.sprite, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)

        blend_premultiplied(frame[y0:y1, x0:x1], warped, self.alpha)
