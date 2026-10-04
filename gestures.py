"""Распознавание жестов: правила по точкам руки, дебаунс, данные о руке.

Модуль не зависит от MediaPipe и OpenCV: он работает с любыми объектами,
у которых есть атрибуты x, y, z (точки руки), поэтому легко тестируется.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ----------------------------- настройки ------------------------------------

HOLD_FRAMES = 5       # сколько кадров подряд жест должен держаться (дебаунс)
MIN_SCORE = 0.6       # минимальная уверенность модели MediaPipe

# «Пистолет». Все расстояния считаются от запястья (точка 0) в 3D.
GUN_INDEX_EXT = 1.45  # вытянутый палец: tip дальше от запястья, чем его MCP, в N раз
GUN_CURL = 1.3        # согнутый палец: tip ближе, чем N * MCP
GUN_THUMB = 0.4       # большой отведён: dist(tip, MCP указательного) > N * длина ладони
GUN_PAIR = 0.45       # вариант «два пальца»: tips указательного и среднего ближе N * ладонь
MIN_AIM_PX = 30       # направление пальца короче этого (px) - палец смотрит в камеру

# «Сердечко» из двух рук. Расстояния в долях длины ладони.
HEART_TOUCH = 0.7     # кончики больших и кончики указательных почти касаются
HEART_HEIGHT = 0.3    # большие пальцы ниже указательных не меньше чем на N
HEART_WRIST = 0.9     # запястья разведены (иначе это «молитва», а не сердце)


# ----------------------------- рука -----------------------------------------

@dataclass(frozen=True)
class Hand:
    """Данные о руке в пикселях кадра, нужные эффектам."""
    palm: tuple[float, float]               # центр ладони
    tip: tuple[float, float]                # кончик указательного пальца
    aim: tuple[float, float] | None         # единичный вектор «куда смотрит палец»
    frame_size: tuple[int, int]             # (ширина, высота) кадра


def make_hand(lms, w: int, h: int) -> Hand:
    """Собирает Hand из нормализованных точек MediaPipe."""
    palm = ((lms[0].x + lms[9].x) / 2 * w, (lms[0].y + lms[9].y) / 2 * h)
    tip = (lms[8].x * w, lms[8].y * h)
    vx = tip[0] - lms[5].x * w
    vy = tip[1] - lms[5].y * h
    length = math.hypot(vx, vy)
    aim = (vx / length, vy / length) if length > MIN_AIM_PX else None
    return Hand(palm, tip, aim, (w, h))


# ----------------------------- правила жестов -------------------------------

def _dist(p, q) -> float:
    return math.sqrt((p.x - q.x) ** 2 + (p.y - q.y) ** 2 + (p.z - q.z) ** 2)


def is_gun(pts) -> bool:
    """Жест «пистолет», две вариации:

    1. указательный вытянут, большой отведён, остальные согнуты;
    2. указательный и средний вытянуты и прижаты друг к другу («два пальца»),
       большой отведён, безымянный и мизинец согнуты.

    Расстояния считаются в 3D (лучше передавать hand_world_landmarks), поэтому
    жест узнаётся и тогда, когда палец направлен прямо в камеру. От Victory
    вариацию 2 отличают сомкнутые пальцы и отведённый большой палец.
    """
    if not pts:
        return False

    def d(a: int, b: int) -> float:
        return _dist(pts[a], pts[b])

    size = d(0, 9)  # длина ладони
    if size < 1e-6:
        return False
    if d(8, 0) <= d(5, 0) * GUN_INDEX_EXT:              # указательный вытянут
        return False
    if d(4, 5) <= size * GUN_THUMB:                     # большой отведён
        return False
    if d(16, 0) >= d(13, 0) * GUN_CURL or d(20, 0) >= d(17, 0) * GUN_CURL:
        return False                                    # безымянный и мизинец согнуты
    if d(12, 0) < d(9, 0) * GUN_CURL:                   # вариация 1: средний согнут
        return True
    return d(12, 0) > d(9, 0) * GUN_INDEX_EXT and d(8, 12) < size * GUN_PAIR


def is_heart(a, b, w: int, h: int) -> bool:
    """Сердечко двумя руками: кончики указательных сходятся сверху, кончики
    больших - снизу, запястья разведены. a и b - нормализованные точки рук."""
    def pt(lms, i):
        return lms[i].x * w, lms[i].y * h

    size = (math.dist(pt(a, 0), pt(a, 9)) + math.dist(pt(b, 0), pt(b, 9))) / 2
    if size < 1:
        return False
    thumb_a, thumb_b = pt(a, 4), pt(b, 4)
    index_a, index_b = pt(a, 8), pt(b, 8)
    if math.dist(thumb_a, thumb_b) > size * HEART_TOUCH:
        return False
    if math.dist(index_a, index_b) > size * HEART_TOUCH:
        return False
    if math.dist(pt(a, 0), pt(b, 0)) < size * HEART_WRIST:
        return False
    thumbs_y = (thumb_a[1] + thumb_b[1]) / 2
    indexes_y = (index_a[1] + index_b[1]) / 2
    return thumbs_y - indexes_y > size * HEART_HEIGHT


def classify(categories, pts) -> str | None:
    """Имя жеста для одной руки. Собственное правило «пистолет» приоритетнее
    модели, иначе его перебивают Victory / Pointing_Up. categories - список
    категорий MediaPipe для этой руки (лучшая первая)."""
    if is_gun(pts):
        return "Gun"
    if categories and categories[0].score >= MIN_SCORE:
        return categories[0].category_name
    return None


# ----------------------------- дебаунс --------------------------------------

class Debouncer:
    """Решает, когда жест «сработал»: он должен продержаться hold_frames кадров
    подряд, а между срабатываниями одного жеста одной руки - пройти cooldown."""

    def __init__(self, cooldown: dict[str, float], hold_frames: int = HOLD_FRAMES):
        self.cooldown = cooldown
        self.hold_frames = hold_frames
        self._state: dict[int, list] = {}                  # рука -> [жест, кадров подряд]
        self._last: dict[tuple[int, str], float] = {}      # (рука, жест) -> время

    def update(self, hand: int, name: str | None, now: float) -> bool:
        st = self._state.setdefault(hand, [None, 0])
        if name == st[0]:
            st[1] += 1
        else:
            st[0], st[1] = name, 1
        if name not in self.cooldown or st[1] < self.hold_frames:
            return False
        key = (hand, name)
        if now - self._last.get(key, -math.inf) < self.cooldown[name]:
            return False
        self._last[key] = now
        return True

    def prune(self, hands_visible: int) -> None:
        """Забыть руки, которых больше нет в кадре."""
        for hand in [k for k in self._state if k >= hands_visible]:
            del self._state[hand]
