import wave

import sound


def test_gunshot_wav_is_valid_mono_16bit(tmp_path):
    path = tmp_path / "shot.wav"
    sound.make_gunshot_wav(path)
    with wave.open(str(path), "rb") as f:
        assert f.getnchannels() == 1 and f.getsampwidth() == 2
        assert 0.4 < f.getnframes() / f.getframerate() < 0.6


def test_sound_without_player_is_silent_noop(monkeypatch):
    monkeypatch.setattr(sound.shutil, "which", lambda name: None)
    monkeypatch.setattr(sound.sys, "platform", "linux")
    s = sound.Sound()
    assert s.mode is None
    s.play()  # не должно падать
