import numpy as np
import pytest

import effects
from gestures import Hand

SIZE = (960, 540)


def hand(aim=(0.8, -0.6), tip=(400.0, 300.0)):
    return Hand(palm=(380.0, 340.0), tip=tip, aim=aim, frame_size=SIZE)


class FakeSound:
    def __init__(self):
        self.plays = 0

    def play(self):
        self.plays += 1


def run(fx, seconds, dt=1 / 30):
    for _ in range(int(seconds / dt)):
        fx.update(dt)


def total_objects(fx):
    return sum(map(len, (fx.particles, fx.rings, fx.flashes, fx.tracers, fx.rockets, fx.popups)))


# ---------------------------- ray_to_edge -----------------------------------

@pytest.mark.parametrize("args,expected", [
    ((100, 100, 1, 0, 960, 540), 860),
    ((100, 100, -1, 0, 960, 540), 100),
    ((100, 100, 0, 1, 960, 540), 440),
    ((100, 100, 0, -1, 960, 540), 100),
    ((0, 0, 0, 0, 960, 540), 0),
])
def test_ray_to_edge_axes(args, expected):
    assert effects.ray_to_edge(*args) == pytest.approx(expected)


def test_ray_to_edge_diagonal_hits_nearest_wall():
    # вверх-вправо из (400, 300): раньше всего упрёмся в верх (y=0) за 300/0.6 = 500
    assert effects.ray_to_edge(400, 300, 0.8, -0.6, *SIZE) == pytest.approx(500)


# ---------------------------- эффекты ---------------------------------------

@pytest.mark.parametrize("name", ["confetti", "explosion", "firework", "gunshot"])
def test_every_gesture_effect_spawns_and_dies(name):
    fx = effects.FX()
    getattr(fx, name)(hand())
    assert total_objects(fx) > 0
    run(fx, 6.0)
    assert total_objects(fx) == 0
    assert fx.shake < 0.5


def test_gunshot_with_aim_spawns_tracer_and_plays_sound():
    snd = FakeSound()
    fx = effects.FX(snd)
    fx.gunshot(hand())
    assert len(fx.tracers) == 1 and snd.plays == 1
    assert 0 < fx.shake < 24


def test_gunshot_at_camera_has_no_tracer_but_bigger_shake():
    fx = effects.FX(FakeSound())
    fx.gunshot(hand(aim=None))
    assert not fx.tracers
    assert fx.shake == 24


def test_tracer_hits_edge_and_spawns_impact():
    fx = effects.FX()
    fx.gunshot(hand())
    tracer = fx.tracers[0]
    for lst in (fx.particles, fx.flashes, fx.rings, fx.popups):
        lst.clear()
    for _ in range(30):                       # 500 px при 3800 px/с - около 8 кадров
        fx.update(1 / 60)
        if tracer.hit:
            break
    assert tracer.hit
    assert fx.flashes and fx.rings and fx.particles   # вспышка, кольцо и искры попадания


def test_mute_means_no_sound_object_calls():
    effects.FX(None).gunshot(hand())  # не должно падать без звука


def test_puff_spawns_ring_particles_and_text():
    fx = effects.FX()
    fx.puff(300, 200, 120)
    assert fx.rings and fx.particles and fx.popups


# ---------------------------- render ----------------------------------------

def test_render_idle_is_noop():
    fx = effects.FX()
    frame = np.full((540, 960, 3), 17, np.uint8)
    out = fx.render(frame)
    assert out is frame and (frame == 17).all()


def test_render_adds_light_and_keeps_shape():
    fx = effects.FX()
    fx.gunshot(hand())
    frame = np.zeros((540, 960, 3), np.uint8)
    out = fx.render(frame)
    assert out.shape == (540, 960, 3) and out.dtype == np.uint8
    assert out.max() > 0


def test_render_reuses_layer_buffer():
    fx = effects.FX()
    fx.confetti(hand())
    fx.render(np.zeros((540, 960, 3), np.uint8))
    first = fx._layer
    fx.render(np.zeros((540, 960, 3), np.uint8))
    assert fx._layer is first
    fx.render(np.zeros((300, 400, 3), np.uint8))  # другой размер -> новый слой, без падения
    assert fx._layer.shape == (300, 400, 3)


def test_render_never_overflows_on_bright_frame():
    fx = effects.FX()
    fx.explosion(hand())
    frame = np.full((540, 960, 3), 250, np.uint8)
    assert fx.render(frame).max() == 255  # насыщение, а не переполнение (не 0..5)
    assert fx.render(np.full((540, 960, 3), 250, np.uint8)).min() >= 250


def test_shake_only_frame_still_renders():
    fx = effects.FX()
    fx.shake = 10
    out = fx.render(np.zeros((100, 120, 3), np.uint8))
    assert out.shape == (100, 120, 3)


def test_shake_shifts_frame_by_integer_offset_and_replicates_edges(monkeypatch):
    h, w = 40, 60
    frame = np.arange(h * w * 3, dtype=np.uint32).reshape(h, w, 3).astype(np.uint8)
    values = iter([0.5, -0.3])                      # dx = round(5) = 5, dy = round(-3) = -3
    monkeypatch.setattr(effects.random, "uniform", lambda a, b: next(values))
    out = effects._shake(frame, 10)
    assert out.shape == frame.shape and out.dtype == np.uint8 and out.flags["C_CONTIGUOUS"]
    dx, dy = 5, -3
    ys = np.clip(np.arange(h) - dy, 0, h - 1)       # out[y, x] = frame[clamp(y - dy), clamp(x - dx)]
    xs = np.clip(np.arange(w) - dx, 0, w - 1)
    assert (out == frame[ys][:, xs]).all()
