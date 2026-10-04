import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import Pt, build_hand, heart_pair, jitter, rng

import gestures as g

DATA = Path(__file__).parent / "data" / "gun_variant2_landmarks.json"


# ---------------------------- is_gun ----------------------------------------

@pytest.mark.parametrize("kind", ["world", "norm"])
def test_real_hand_gun_variant2(kind):
    """Реальная рука (скриншот пользователя): два пальца вместе + отведённый большой."""
    pts = [Pt(*p) for p in json.loads(DATA.read_text())[kind]]
    assert g.is_gun(pts)


def test_gun_variant1_index_only():
    assert g.is_gun(build_hand(index="ext", middle="curl"))


def test_gun_variant2_two_fingers_together():
    assert g.is_gun(build_hand(index="ext", middle="ext", middle_tip_x=0.0))


def test_gun_works_when_curled_fingers_are_loose():
    # в реальной руке согнутые пальцы не прижаты идеально (ratio ~1.15)
    assert g.is_gun(build_hand(curl_tip_y=1.17))


@pytest.mark.parametrize("pose", [
    dict(index="curl", middle="curl", thumb="tucked"),            # кулак
    dict(index="curl", middle="curl", thumb="out"),               # thumbs up
    dict(index="ext", middle="ext", ring="ext", pinky="ext"),     # открытая ладонь
    dict(index="ext", middle="curl", thumb="tucked"),             # pointing up
    dict(index="ext", middle="ext", thumb="tucked", middle_tip_x=0.6),  # Victory
    dict(index="ext", middle="ext", thumb="out", middle_tip_x=0.8),     # V с отведённым большим
    dict(index="ext", middle="curl", ring="ext", thumb="out"),    # безымянный вытянут
])
def test_not_gun(pose):
    assert not g.is_gun(build_hand(**pose))


def test_gun_degenerate_inputs():
    assert not g.is_gun([])
    assert not g.is_gun([Pt(0, 0, 0)] * 21)  # нулевая длина ладони


@pytest.mark.parametrize("pose,expected", [
    (dict(index="ext", middle="curl"), True),
    (dict(index="ext", middle="ext"), True),
    (dict(index="curl", middle="curl", thumb="tucked"), False),
    (dict(index="ext", middle="ext", ring="ext", pinky="ext"), False),
])
def test_gun_is_stable_under_landmark_noise(pose, expected):
    r = rng(42)
    hits = sum(g.is_gun(jitter(build_hand(**pose), 0.02, r)) for _ in range(300))
    if expected:
        assert hits >= 270   # >= 90% кадров с шумом 2% длины ладони
    else:
        assert hits <= 15    # <= 5% ложных срабатываний


def test_victory_with_close_fingers_but_tucked_thumb_is_not_gun():
    v = build_hand(index="ext", middle="ext", thumb="tucked", middle_tip_x=0.35)
    assert not g.is_gun(v)


# ---------------------------- is_heart --------------------------------------

def test_heart_detected():
    a, b = heart_pair()
    assert g.is_heart(a, b, 1000, 1000)


def test_heart_not_when_hands_apart():
    a, b = heart_pair(a_thumb_tip=(0.2, 0.7), b_thumb_tip=(0.8, 0.7))
    assert not g.is_heart(a, b, 1000, 1000)


def test_heart_not_when_only_thumbs_touch():
    a, b = heart_pair(a_index_tip=(0.3, 0.4), b_index_tip=(0.7, 0.4))
    assert not g.is_heart(a, b, 1000, 1000)


def test_prayer_pose_is_not_heart():
    a, b = heart_pair(a_wrist=(0.48, 0.85), b_wrist=(0.52, 0.85))
    assert not g.is_heart(a, b, 1000, 1000)


def test_inverted_heart_is_not_heart():
    a, b = heart_pair(a_thumb_tip=(0.48, 0.30), b_thumb_tip=(0.52, 0.30))
    assert not g.is_heart(a, b, 1000, 1000)


def test_heart_degenerate_hands():
    zero = [Pt(0, 0, 0)] * 21
    assert not g.is_heart(zero, zero, 1000, 1000)


# ---------------------------- classify / make_hand --------------------------

def cat(name, score):
    return [SimpleNamespace(category_name=name, score=score)]


def test_classify_gun_beats_model():
    assert g.classify(cat("Victory", 0.99), build_hand(middle="ext")) == "Gun"


def test_classify_uses_model_above_threshold():
    fist = build_hand(index="curl", thumb="tucked")
    assert g.classify(cat("Closed_Fist", 0.9), fist) == "Closed_Fist"
    assert g.classify(cat("Closed_Fist", g.MIN_SCORE - 0.01), fist) is None
    assert g.classify([], fist) is None


def lms_for_aim(tip, mcp):
    pts = [Pt(0.5, 0.8, 0)] * 21
    pts[9], pts[5], pts[8] = Pt(0.5, 0.4, 0), Pt(*mcp, 0), Pt(*tip, 0)
    return pts


def test_make_hand_aim_and_palm():
    hand = g.make_hand(lms_for_aim(tip=(0.6, 0.3), mcp=(0.5, 0.4)), 1000, 500)
    assert hand.palm == pytest.approx((500, 300))
    assert hand.tip == pytest.approx((600, 150))
    assert hand.aim == pytest.approx((0.8944, -0.4472), abs=1e-3)  # единичный вектор (100, -50) px
    assert hand.frame_size == (1000, 500)


def test_make_hand_finger_pointing_at_camera_has_no_aim():
    hand = g.make_hand(lms_for_aim(tip=(0.505, 0.395), mcp=(0.5, 0.4)), 1000, 500)
    assert hand.aim is None


# ---------------------------- Debouncer -------------------------------------

def make_db(hold=3, cd=1.0):
    return g.Debouncer({"Gun": cd, "Open_Palm": cd}, hold_frames=hold)


def test_debounce_fires_only_after_hold_frames():
    db = make_db()
    assert [db.update(0, "Gun", t) for t in (0.0, 0.1, 0.2)] == [False, False, True]


def test_debounce_cooldown_then_refire():
    db = make_db()
    for t in (0.0, 0.1, 0.2):
        db.update(0, "Gun", t)
    assert not db.update(0, "Gun", 0.5)   # внутри паузы
    assert db.update(0, "Gun", 1.3)       # пауза прошла, жест всё ещё держится


def test_debounce_resets_on_gesture_change_and_none():
    db = make_db()
    db.update(0, "Gun", 0.0); db.update(0, "Gun", 0.1)
    db.update(0, None, 0.2)
    assert not db.update(0, "Gun", 0.3)   # счёт начался заново
    db.update(0, "Open_Palm", 0.4)
    assert not db.update(0, "Gun", 0.5)


def test_debounce_ignores_unknown_gestures():
    db = make_db()
    assert not any(db.update(0, "Thumb_Up", t / 10) for t in range(20))


def test_debounce_hands_are_independent_and_pruned():
    db = make_db()
    for t in (0.0, 0.1):
        db.update(0, "Gun", t); db.update(1, "Gun", t)
    assert db.update(0, "Gun", 0.2) and db.update(1, "Gun", 0.2)
    db.prune(1)
    assert 1 not in db._state and 0 in db._state
