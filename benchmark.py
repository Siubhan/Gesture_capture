"""Замеры производительности частей, из которых состоит кадр.

    python benchmark.py [--n 300]

Скрипт не использует камеру и MediaPipe: он меряет только нашу часть (жесты,
эффекты, замена лица). Время самого распознавания рук и лиц на вашей машине
покажет `python gesture_fx.py --profile`.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import faceswap as fs
from effects import FX
from gestures import Debouncer, Hand, is_gun, is_heart, make_hand

W, H = 960, 540
FIXTURE = Path(__file__).parent / "tests" / "data" / "gun_variant2_landmarks.json"


def bench(fn, n: int) -> float:
    """Секунд на вызов: медиана из 5 прогонов по n вызовов."""
    fn()  # прогрев
    runs = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(n):
            fn()
        runs.append((time.perf_counter() - t) / n)
    return statistics.median(runs)


def heavy_scene() -> FX:
    random.seed(1)
    fx = FX()
    hand = Hand(palm=(480, 300), tip=(500, 250), aim=(0.6, -0.8), frame_size=(W, H))
    fx.explosion(hand)
    fx.confetti(hand)
    fx.firework(hand)
    fx.gunshot(hand)
    for _ in range(3):
        fx.update(1 / 60)
    return fx


def detection(eye: float):
    kp = [SimpleNamespace(x=(480 - eye / 2) / W, y=270 / H),
          SimpleNamespace(x=(480 + eye / 2) / W, y=270 / H)]
    return SimpleNamespace(keypoints=kp, bounding_box=SimpleNamespace(width=eye * 2.3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="число вызовов в одном прогоне")
    n = ap.parse_args().n

    rows: list[tuple[str, float]] = []

    def add(name, seconds):
        rows.append((name, seconds * 1000))

    # --- жесты ---
    data = json.loads(FIXTURE.read_text())
    world = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in data["world"]]
    norm = [SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in data["norm"]]
    debounce = Debouncer({"Gun": 0.7})
    add("is_gun (на реальной руке)", bench(lambda: is_gun(world), n * 20))
    add("is_heart (две руки)", bench(lambda: is_heart(norm, norm, W, H), n * 20))
    add("make_hand", bench(lambda: make_hand(norm, W, H), n * 20))
    add("Debouncer.update", bench(lambda: debounce.update(0, "Gun", 1.0), n * 20))

    # --- эффекты ---
    frame = np.zeros((H, W, 3), np.uint8)
    idle = FX()
    add("FX.render, пусто", bench(lambda: idle.render(frame), n))
    heavy = heavy_scene()
    count = len(heavy.particles) + len(heavy.rings) + len(heavy.flashes)
    add(f"FX.render, {count} объектов", bench(lambda: heavy.render(frame), n))
    add(f"FX.update, {count} объектов", bench(lambda: heavy.update(1 / 600), n))

    # --- замена лица ---
    sprite = fs.FaceSwap.load()
    if sprite is None:
        print("assets/pepe.png не найден: замеры замены лица пропущены")
    else:
        sprite.alpha = 1.0
        for label, eye in (("лицо далеко", 40), ("лицо обычное", 80), ("лицо близко", 160)):
            det = [detection(eye)]
            add(f"FaceSwap.apply, {label} (глаза {eye}px)",
                bench(lambda det=det: sprite.apply(frame.copy(), det, W, H), n))

    width = max(len(name) for name, _ in rows)
    print(f"\nКадр {W}x{H}, {n} вызовов x 5 прогонов, медиана\n")
    for name, ms in rows:
        shown = f"{ms * 1000:8.1f} мкс" if ms < 0.1 else f"{ms:8.2f} мс "
        print(f"  {name:<{width}}  {shown}")

    by = dict(rows)
    ours = (by.get(f"FX.render, {count} объектов", 0) + by.get(f"FX.update, {count} объектов", 0)
            + by.get("FaceSwap.apply, лицо обычное (глаза 80px)", 0))
    print(f"\nНаша часть кадра в худшем случае (эффекты + жаба): {ours:.2f} мс")
    print("Бюджет кадра: 33.3 мс при 30 FPS, 16.7 мс при 60 FPS")


if __name__ == "__main__":
    main()
