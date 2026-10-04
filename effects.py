"""Визуальные эффекты: частицы, вспышки, трассеры и менеджер FX.

Все эффекты рисуются на отдельном чёрном слое, который добавляется к кадру
(аддитивно) вместе с размытым свечением.
"""

from __future__ import annotations

import math
import random
from itertools import chain

import cv2
import numpy as np

PALETTE = [
    (80, 80, 255), (80, 200, 255), (80, 255, 120),
    (255, 200, 60), (255, 90, 200), (255, 255, 255),
]  # BGR

GLOW_SCALE = 4     # свечение считается на уменьшенной в N раз копии слоя
GLOW_SIGMA = 2.0   # размытие на уменьшенной копии (на полном кадре ~ 8 px)


def _fade(color, k):
    """Цвет, затухающий с коэффициентом k (слой аддитивный: тёмный = прозрачный)."""
    return tuple(int(v * k) for v in color)


# ----------------------------- объекты эффектов -----------------------------

class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color",
                 "size", "gravity", "drag", "shape")

    def __init__(self, x, y, vx, vy, life, color, size,
                 gravity=0.0, drag=1.0, shape="circle"):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color, self.size = color, size
        self.gravity, self.drag, self.shape = gravity, drag, shape

    def update(self, dt):
        damping = self.drag ** (dt * 60)
        self.vx *= damping
        self.vy *= damping
        self.vy += self.gravity * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.life -= dt

    def draw(self, layer):
        k = max(self.life / self.max_life, 0.0)
        c = _fade(self.color, k)
        x, y = int(self.x), int(self.y)
        if self.shape == "rect":
            s = max(int(self.size), 1)
            cv2.rectangle(layer, (x - s, y - s // 2), (x + s, y + s // 2), c, -1)
        else:
            cv2.circle(layer, (x, y), max(int(self.size * (0.4 + 0.6 * k)), 1), c, -1)


class Ring:
    """Расширяющееся кольцо (ударная волна)."""

    def __init__(self, x, y, max_r, duration, color, thickness=6):
        self.x, self.y, self.max_r = x, y, max_r
        self.life = self.max_life = duration
        self.color, self.thickness = color, thickness

    def update(self, dt):
        self.life -= dt

    def draw(self, layer):
        t = 1.0 - max(self.life / self.max_life, 0.0)
        r = int(self.max_r * (1 - (1 - t) ** 3))  # ease-out
        k = 1.0 - t
        cv2.circle(layer, (int(self.x), int(self.y)), max(r, 1),
                   _fade(self.color, k), max(int(self.thickness * k), 1))


class Flash:
    """Короткая яркая вспышка (дульное пламя)."""

    def __init__(self, x, y, r, duration, color):
        self.x, self.y, self.r, self.color = x, y, r, color
        self.life = self.max_life = duration

    def update(self, dt):
        self.life -= dt

    def draw(self, layer):
        k = max(self.life / self.max_life, 0.0)
        cv2.circle(layer, (int(self.x), int(self.y)),
                   max(int(self.r * (0.5 + 0.5 * k)), 1), _fade(self.color, k), -1)


class Popup:
    """Всплывающая надпись, которая поднимается и гаснет."""

    def __init__(self, x, y, text, color, duration=0.7, scale=1.5):
        self.x, self.y, self.text, self.color, self.scale = x, y, text, color, scale
        self.life = self.max_life = duration
        (self._width, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)

    def update(self, dt):
        self.life -= dt
        self.y -= 70 * dt

    def draw(self, layer):
        k = max(self.life / self.max_life, 0.0)
        cv2.putText(layer, self.text, (int(self.x - self._width / 2), int(self.y)),
                    cv2.FONT_HERSHEY_DUPLEX, self.scale, _fade(self.color, k), 3,
                    cv2.LINE_AA)


class Tracer:
    """Пуля-трассер: быстрая светящаяся линия вдоль направления пальца."""

    SPEED = 3800.0
    LENGTH = 170.0

    def __init__(self, x, y, dx, dy, dist):
        self.x0, self.y0, self.dx, self.dy, self.dist = x, y, dx, dy, dist
        self.s = 0.0
        self.hit = False

    @property
    def alive(self):
        return self.s - self.LENGTH < self.dist

    def update(self, dt, fx):
        self.s += self.SPEED * dt
        if self.s >= self.dist and not self.hit:
            self.hit = True
            fx.impact(self.x0 + self.dx * self.dist,
                      self.y0 + self.dy * self.dist, (self.dx, self.dy))

    def draw(self, layer):
        head = min(self.s, self.dist)
        tail = min(max(self.s - self.LENGTH, 0.0), self.dist)
        if tail >= head:
            return
        p1 = (int(self.x0 + self.dx * tail), int(self.y0 + self.dy * tail))
        p2 = (int(self.x0 + self.dx * head), int(self.y0 + self.dy * head))
        cv2.line(layer, p1, p2, (90, 190, 255), 3, cv2.LINE_AA)
        cv2.line(layer, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)


class Rocket:
    """Ракета фейерверка: летит вверх, потом взрывается."""

    def __init__(self, x, y, target_y, color):
        self.x, self.y, self.target_y, self.color = x, y, target_y, color
        self.vy = -650.0
        self.life = 5.0
        self.exploded = False

    def update(self, dt, fx):
        self.y += self.vy * dt
        self.life -= dt
        fx.particles.append(Particle(
            self.x + random.uniform(-2, 2), self.y, random.uniform(-15, 15),
            random.uniform(20, 60), 0.35, (180, 220, 255), 3, drag=0.98))
        if self.y <= self.target_y and not self.exploded:
            self.exploded = True
            fx.firework_burst(self.x, self.y, self.color)

    def draw(self, layer):
        cv2.circle(layer, (int(self.x), int(self.y)), 4, (255, 255, 255), -1)


def ray_to_edge(x, y, dx, dy, w, h):
    """Расстояние от точки (x, y) вдоль (dx, dy) до границы кадра."""
    ts = []
    if dx > 1e-6:
        ts.append((w - x) / dx)
    elif dx < -1e-6:
        ts.append((0 - x) / dx)
    if dy > 1e-6:
        ts.append((h - y) / dy)
    elif dy < -1e-6:
        ts.append((0 - y) / dy)
    return max(min(ts), 0.0) if ts else 0.0


def _shake(frame, amplitude):
    """Сдвигает кадр на случайное целое число пикселей (края растягиваются).
    Целочисленный сдвиг через copyMakeBorder в разы быстрее warpAffine."""
    h, w = frame.shape[:2]
    pad = int(amplitude) + 1
    dx = round(random.uniform(-1, 1) * amplitude)
    dy = round(random.uniform(-1, 1) * amplitude)
    padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    return np.ascontiguousarray(padded[pad - dy:pad - dy + h, pad - dx:pad - dx + w])


# ----------------------------- менеджер -------------------------------------

class FX:
    """Хранит активные эффекты и умеет их запускать.

    Эффекты-жесты принимают один аргумент hand (gestures.Hand).
    """

    def __init__(self, sound=None):
        self.particles, self.rings, self.rockets = [], [], []
        self.flashes, self.tracers, self.popups = [], [], []
        self.shake = 0.0
        self.sound = sound
        self._layer = None  # переиспользуемый слой, чтобы не выделять память каждый кадр

    # --- 1. конфетти ---
    def confetti(self, hand):
        x, y = hand.palm
        for _ in range(140):
            a = random.uniform(-math.pi, 0)  # вверх веером
            sp = random.uniform(150, 650)
            self.particles.append(Particle(
                x, y, math.cos(a) * sp, math.sin(a) * sp,
                random.uniform(1.6, 2.8), random.choice(PALETTE),
                random.uniform(3, 7), gravity=700, drag=0.985, shape="rect"))

    # --- 2. взрыв ---
    def explosion(self, hand):
        x, y = hand.palm
        self.rings.append(Ring(x, y, 260, 0.6, (60, 140, 255), 10))
        self.rings.append(Ring(x, y, 170, 0.45, (255, 255, 255), 5))
        for _ in range(180):
            a = random.uniform(0, 2 * math.pi)
            sp = random.uniform(100, 900)
            col = random.choice([(40, 100, 255), (60, 180, 255), (200, 240, 255)])
            self.particles.append(Particle(
                x, y, math.cos(a) * sp, math.sin(a) * sp,
                random.uniform(0.4, 1.1), col, random.uniform(2, 6),
                gravity=200, drag=0.93))

    # --- 3. фейерверк ---
    def firework(self, hand):
        x, y = hand.palm
        for _ in range(3):
            tx = x + random.uniform(-160, 160)
            ty = max(60, y - random.uniform(220, 380))
            self.rockets.append(Rocket(tx, y, ty, random.choice(PALETTE)))

    def firework_burst(self, x, y, color):
        n = 90
        for i in range(n):
            a = 2 * math.pi * i / n
            sp = random.choice([260, 340]) * random.uniform(0.9, 1.1)
            self.particles.append(Particle(
                x, y, math.cos(a) * sp, math.sin(a) * sp,
                random.uniform(1.0, 1.6), color, 3.5,
                gravity=180, drag=0.95))
        self.rings.append(Ring(x, y, 60, 0.3, color, 3))

    # --- 4. выстрел ---
    def gunshot(self, hand):
        tx, ty = hand.tip
        w, h = hand.frame_size
        if self.sound:
            self.sound.play()

        flame = [(40, 150, 255), (80, 220, 255), (255, 255, 255)]
        if hand.aim is None:
            # палец направлен прямо в камеру: большая вспышка во весь экран
            self.flashes.append(Flash(tx, ty, 100, 0.16, (110, 190, 255)))
            self.flashes.append(Flash(tx, ty, 50, 0.11, (255, 255, 255)))
            self.rings.append(Ring(tx, ty, 240, 0.4, (180, 230, 255), 8))
            for _ in range(70):
                a = random.uniform(0, 2 * math.pi)
                sp = random.uniform(200, 900)
                self.particles.append(Particle(
                    tx, ty, math.cos(a) * sp, math.sin(a) * sp,
                    random.uniform(0.15, 0.4), random.choice(flame),
                    random.uniform(2, 5), drag=0.92))
            self.shake = 24
        else:
            dx, dy = hand.aim
            mx, my = tx + dx * 8, ty + dy * 8
            self.flashes.append(Flash(mx, my, 55, 0.12, (110, 190, 255)))
            self.flashes.append(Flash(mx, my, 26, 0.08, (255, 255, 255)))
            base = math.atan2(dy, dx)
            for _ in range(32):  # конус искр вдоль ствола
                a = base + random.gauss(0, 0.28)
                sp = random.uniform(300, 1000)
                self.particles.append(Particle(
                    mx, my, math.cos(a) * sp, math.sin(a) * sp,
                    random.uniform(0.1, 0.3), random.choice(flame),
                    random.uniform(2, 4), drag=0.9))
            for _ in range(6):  # дымок
                self.particles.append(Particle(
                    mx, my, random.uniform(-30, 30), random.uniform(-70, -20),
                    random.uniform(0.7, 1.1), (70, 70, 70),
                    random.uniform(7, 11), drag=0.96))
            self.tracers.append(Tracer(mx, my, dx, dy, ray_to_edge(mx, my, dx, dy, w, h)))
            self.shake = 12
        self.popups.append(Popup(tx, ty - 50, "BANG!", (60, 190, 255)))

    def impact(self, x, y, d):
        """Попадание трассера в край кадра: искры назад и вспышка."""
        back = math.atan2(-d[1], -d[0])
        self.flashes.append(Flash(x, y, 30, 0.12, (150, 220, 255)))
        self.rings.append(Ring(x, y, 50, 0.25, (200, 240, 255), 3))
        for _ in range(36):
            a = back + random.uniform(-1.3, 1.3)
            sp = random.uniform(150, 650)
            self.particles.append(Particle(
                x, y, math.cos(a) * sp, math.sin(a) * sp,
                random.uniform(0.2, 0.6), random.choice(PALETTE[:3] + [(200, 240, 255)]),
                random.uniform(2, 4), gravity=500, drag=0.93))

    # --- 5. облачко при превращении лица в жабу ---
    def puff(self, x, y, radius):
        green = [(90, 220, 120), (60, 180, 90), (220, 255, 230)]
        self.rings.append(Ring(x, y, radius * 1.3, 0.45, green[0], 8))
        for _ in range(70):
            a = random.uniform(0, 2 * math.pi)
            sp = random.uniform(80, 420)
            self.particles.append(Particle(
                x + math.cos(a) * radius * 0.5, y + math.sin(a) * radius * 0.5,
                math.cos(a) * sp, math.sin(a) * sp, random.uniform(0.5, 1.0),
                random.choice(green), random.uniform(4, 9), drag=0.93))
        self.popups.append(Popup(x, y - radius, "RIBBIT!", green[0], duration=0.9))

    # --- цикл ---
    def update(self, dt):
        for r in self.rockets:
            r.update(dt, self)
        for t in self.tracers:
            t.update(dt, self)
        for obj in chain(self.particles, self.rings, self.flashes, self.popups):
            obj.update(dt)
        self.shake *= 0.82 ** (dt * 60)

        self.rockets = [r for r in self.rockets if not r.exploded and r.life > 0]
        self.tracers = [t for t in self.tracers if t.alive]
        self.particles = [p for p in self.particles if p.life > 0]
        self.rings = [r for r in self.rings if r.life > 0]
        self.flashes = [f for f in self.flashes if f.life > 0]
        self.popups = [p for p in self.popups if p.life > 0]

    def _objects(self):
        return chain(self.rings, self.flashes, self.particles, self.tracers,
                     self.rockets, self.popups)

    @property
    def has_objects(self):
        return bool(self.particles or self.rings or self.flashes or self.tracers
                    or self.rockets or self.popups)

    def render(self, frame):
        """Накладывает эффекты на кадр (in-place) и возвращает его.
        Если ничего не происходит, возвращает кадр без лишней работы."""
        if self.has_objects:
            h, w = frame.shape[:2]
            if self._layer is None or self._layer.shape != frame.shape:
                self._layer = np.zeros_like(frame)
            else:
                self._layer.fill(0)
            layer = self._layer
            for obj in self._objects():
                obj.draw(layer)
            # свечение: размываем уменьшенную копию, это в разы дешевле, чем весь кадр
            small = cv2.resize(layer, (w // GLOW_SCALE, h // GLOW_SCALE),
                               interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), GLOW_SIGMA)
            glow = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.add(glow, layer, dst=glow)
            cv2.add(frame, glow, dst=frame)
        if self.shake > 0.5:  # отдача: тряска кадра
            frame = _shake(frame, self.shake)
        return frame
