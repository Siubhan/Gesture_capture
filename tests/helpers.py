"""Построители синтетических рук для тестов жестов."""

import random
from collections import namedtuple

Pt = namedtuple("Pt", "x y z")

# Условная рука: запястье в (0, 0), длина ладони (запястье -> MCP среднего) = 1.
_FINGER_X = {"index": -0.30, "middle": 0.0, "ring": 0.28, "pinky": 0.50}
_IDX = {"index": (5, 8), "middle": (9, 12), "ring": (13, 16), "pinky": (17, 20)}


def build_hand(index="ext", middle="curl", ring="curl", pinky="curl",
               thumb="out", middle_tip_x=0.0, curl_tip_y=1.12):
    """21 точка 3D. Пальцы: 'ext' (вытянут) или 'curl' (согнут).
    thumb: 'out' (отведён) или 'tucked' (прижат). middle_tip_x сдвигает
    кончик среднего пальца по x (раздвигает «V»)."""
    pts = [Pt(0.0, 0.0, 0.0)] * 21
    states = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, (mcp, tip) in _IDX.items():
        x = _FINGER_X[name]
        pts[mcp] = Pt(x, 1.0, 0.0)
        tip_x = x + (middle_tip_x if name == "middle" else 0.0)
        pts[tip] = Pt(tip_x, 1.9 if states[name] == "ext" else curl_tip_y, 0.0)
    pts[9] = Pt(0.0, 1.0, 0.0)  # MCP среднего: ровно длина ладони
    pts[4] = Pt(-1.0, 0.9, 0.0) if thumb == "out" else Pt(-0.15, 0.7, 0.0)
    return pts


def jitter(pts, sigma, rng):
    return [Pt(p.x + rng.gauss(0, sigma), p.y + rng.gauss(0, sigma),
               p.z + rng.gauss(0, sigma)) for p in pts]


def norm_hand(wrist, mcp9, thumb_tip, index_tip):
    """Нормализованная рука (0..1) для is_heart: заполнены только нужные точки."""
    pts = [Pt(0.0, 0.0, 0.0)] * 21
    pts[0], pts[9] = Pt(*wrist, 0.0), Pt(*mcp9, 0.0)
    pts[4], pts[8] = Pt(*thumb_tip, 0.0), Pt(*index_tip, 0.0)
    return pts


def rng(seed=1):
    return random.Random(seed)


def heart_pair(**over):
    a = dict(wrist=(0.30, 0.85), mcp9=(0.38, 0.58), thumb_tip=(0.48, 0.70), index_tip=(0.49, 0.40))
    b = dict(wrist=(0.70, 0.85), mcp9=(0.62, 0.58), thumb_tip=(0.52, 0.70), index_tip=(0.51, 0.40))
    for k, v in over.items():
        side, key = k.split("_", 1)
        (a if side == "a" else b)[key] = v
    return norm_hand(**a), norm_hand(**b)
