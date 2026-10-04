import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import faceswap as fs

W, H = 640, 480


def detection(left=(250, 220), right=(330, 220), box_w=180, order="rl"):
    """Заглушка детекции MediaPipe: точки 0 и 1 - глаза в пикселях кадра."""
    kps = [SimpleNamespace(x=left[0] / W, y=left[1] / H),
           SimpleNamespace(x=right[0] / W, y=right[1] / H)]
    if order == "lr":  # у MediaPipe «правый глаз» субъекта идёт первым, слева в кадре или нет
        kps.reverse()
    return SimpleNamespace(keypoints=kps, bounding_box=SimpleNamespace(width=box_w))


@pytest.fixture
def swap():
    sw = fs.FaceSwap.load()
    assert sw is not None, "assets/pepe.png должен существовать"
    return sw


# ---------------------------- спрайт ----------------------------------------

def test_sprite_asset_has_transparency_and_valid_eye_anchors():
    img = cv2.imdecode(np.fromfile(str(fs.SPRITE_PATH), np.uint8), cv2.IMREAD_UNCHANGED)
    assert img.ndim == 3 and img.shape[2] == 4
    alpha = img[:, :, 3]
    assert alpha.min() == 0 and alpha.max() == 255
    for x, y in fs.SPRITE_EYES:
        assert 0 <= x < img.shape[1] and 0 <= y < img.shape[0]
        assert alpha[int(y), int(x)] == 255, "глаза должны лежать внутри непрозрачной области"


def test_load_missing_or_non_alpha_sprite_returns_none(tmp_path):
    assert fs.FaceSwap.load(tmp_path / "nope.png") is None
    cv2.imencode(".png", np.zeros((10, 10, 3), np.uint8))[1].tofile(str(tmp_path / "rgb.png"))
    assert fs.FaceSwap.load(tmp_path / "rgb.png") is None


# ---------------------------- детекции --------------------------------------

def test_face_from_detection_geometry_and_eye_order():
    for order in ("rl", "lr"):
        f = fs.face_from_detection(detection(order=order), W, H)
        assert (f.cx, f.cy) == pytest.approx((290, 220))
        assert f.eye == pytest.approx(80) and f.box_w == 180
        assert f.angle == pytest.approx(0)


def test_face_from_detection_tilt_and_degenerate():
    f = fs.face_from_detection(detection(left=(250, 200), right=(330, 240)), W, H)
    assert f.angle == pytest.approx(math.atan2(40, 80))
    assert fs.face_from_detection(SimpleNamespace(keypoints=[], bounding_box=None), W, H) is None
    assert fs.face_from_detection(detection(left=(100, 100), right=(100, 100)), W, H) is None


# ---------------------------- состояние -------------------------------------

def test_fade_in_hold_and_fade_out(swap):
    t = 0.0
    swap.trigger(t)
    assert swap.puff_pending and not swap.visible
    for _ in range(10):                      # 10 * 0.05 = 0.5 c > FADE_IN
        swap.trigger(t); swap.update(t, 0.05); t += 0.05
    assert swap.alpha == 1.0
    while t < 0.5 + fs.LINGER - 0.1:         # сердечко убрали, жаба ещё держится
        swap.update(t, 0.05); t += 0.05
    assert swap.alpha == 1.0
    while t < 0.5 + fs.LINGER + fs.FADE_OUT + 0.2:
        swap.update(t, 0.05); t += 0.05
    assert swap.alpha == 0.0 and not swap.visible


def test_puff_only_on_fresh_start_not_while_holding(swap):
    swap.trigger(0.0); swap.update(0.0, 0.05)
    swap.puff_pending = False
    swap.trigger(0.3)
    assert not swap.puff_pending


# ---------------------------- отрисовка -------------------------------------

def gray_frame():
    return np.full((H, W, 3), 128, np.uint8)


def test_apply_changes_face_region_only(swap):
    swap.alpha = 1.0
    frame = gray_frame()
    faces = swap.apply(frame, [detection()], W, H)
    assert len(faces) == 1
    assert (frame != 128).any()
    assert (frame[:40] == 128).all() and (frame[-40:] == 128).all()   # вдали от лица не тронуто
    assert (frame[:, :40] == 128).all() and (frame[:, -40:] == 128).all()


def test_apply_with_zero_alpha_draws_nothing(swap):
    swap.alpha = 0.0
    frame = gray_frame()
    swap.apply(frame, [detection()], W, H)
    assert (frame == 128).all()


def test_partial_alpha_blends_between_frame_and_frog(swap):
    full, half = gray_frame(), gray_frame()
    swap.alpha = 1.0
    swap.apply(full, [detection()], W, H)
    swap._faces = []
    swap.alpha = 0.5
    swap.apply(half, [detection()], W, H)
    d_full = np.abs(full.astype(int) - 128).mean()
    d_half = np.abs(half.astype(int) - 128).mean()
    assert 0.4 < d_half / d_full < 0.6        # половинная прозрачность = половина эффекта


def test_apply_handles_faces_outside_and_on_edges(swap):
    swap.alpha = 1.0
    swap.apply(gray_frame(), [detection(left=(-200, 100), right=(-120, 100))], W, H)
    swap._faces = []
    swap.apply(gray_frame(), [detection(left=(0, 0), right=(60, 5), box_w=100)], W, H)
    swap._faces = []
    swap.apply(gray_frame(), [detection(left=(600, 460), right=(700, 470))], W, H)


def test_apply_tilted_face_rotates_without_crashing(swap):
    swap.alpha = 1.0
    frame = gray_frame()
    swap.apply(frame, [detection(left=(250, 180), right=(330, 260))], W, H)
    assert (frame != 128).any()


def test_two_faces_are_both_covered(swap):
    swap.alpha = 1.0
    frame = gray_frame()
    faces = swap.apply(frame, [detection(left=(100, 220), right=(160, 220), box_w=130),
                               detection(left=(440, 220), right=(500, 220), box_w=130)], W, H)
    assert len(faces) == 2 and faces[0].cx < faces[1].cx
    assert (frame[220, 130] != 128).any() and (frame[220, 470] != 128).any()


def test_detector_dropout_keeps_last_faces_then_clears(swap):
    swap.alpha = 1.0
    swap.apply(gray_frame(), [detection()], W, H)
    for _ in range(fs.MISS_FRAMES):
        assert len(swap.apply(gray_frame(), [], W, H)) == 1
    assert swap.apply(gray_frame(), [], W, H) == []


def test_smoothing_moves_partially_toward_new_position(swap):
    swap.alpha = 1.0
    swap.apply(gray_frame(), [detection()], W, H)                       # cx = 290
    f = swap.apply(gray_frame(), [detection(left=(350, 220), right=(430, 220))], W, H)[0]
    assert 290 < f.cx < 390                                             # новая цель cx = 390


def test_frog_is_scaled_to_cover_the_face(swap):
    swap.alpha = 1.0
    frame = gray_frame()
    swap.apply(frame, [detection(box_w=180)], W, H)
    changed = np.where((frame != 128).any(axis=2))
    width = changed[1].max() - changed[1].min()
    assert width > 180            # жаба шире рамки лица, а не меньше её


@pytest.mark.parametrize("alpha", [1.0, 0.5, 0.2])
def test_blend_matches_float_reference(alpha):
    r = np.random.default_rng(3)
    roi_src = r.integers(0, 256, (50, 70, 3), dtype=np.uint8)
    a = r.integers(0, 256, (50, 70, 1), dtype=np.uint8)
    bgr = (r.integers(0, 256, (50, 70, 3)) * a // 255).astype(np.uint8)   # premultiplied
    warped = np.concatenate([bgr, a], axis=2)

    expected = warped.astype(np.float32) * alpha
    expected = expected[:, :, :3] + roi_src * (1 - expected[:, :, 3:4] / 255.0)

    big = np.zeros((80, 100, 3), np.uint8)
    roi = big[10:60, 20:90]                      # ROI-вид, как во frame[y0:y1, x0:x1]
    roi[:] = roi_src
    fs.blend_premultiplied(roi, warped, alpha)
    assert np.abs(roi.astype(np.float32) - expected).max() <= 2.5
    assert (big[:10] == 0).all() and (big[:, :20] == 0).all()   # за пределами ROI не тронуто
