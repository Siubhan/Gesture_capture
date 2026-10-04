"""Звук выстрела: синтезируется в WAV и проигрывается системными средствами."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

WAV_NAME = "gesture_fx_gunshot_v1.wav"  # версия в имени: при смене звука старый кэш не мешает


def make_gunshot_wav(path, sr=44100):
    """Синтезирует выстрел: резкий щелчок + низкий удар + затухающий шум."""
    n = int(sr * 0.5)
    t = np.arange(n) / sr
    noise = np.random.default_rng(7).uniform(-1, 1, n)
    crack = np.diff(noise, prepend=0.0) * np.exp(-t * 22)
    crack /= np.max(np.abs(crack))
    freq = 45 + 160 * np.exp(-t * 14)
    thump = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t * 9)
    body = noise * np.exp(-t * 7) * 0.25
    sig = 0.55 * crack + 0.9 * thump + body
    sig = 0.9 * sig / np.max(np.abs(sig))
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((sig * 32767).astype(np.int16).tobytes())


class Sound:
    """Windows - winsound, macOS - afplay, Linux - paplay/aplay.
    Если подходящего плеера нет, play() ничего не делает."""

    def __init__(self):
        self.mode = None
        self.cmd = None
        self.path = os.path.join(tempfile.gettempdir(), WAV_NAME)
        try:
            if not os.path.exists(self.path):
                make_gunshot_wav(self.path)
            if sys.platform == "win32":
                import winsound
                self._winsound = winsound
                self.mode = "winsound"
            else:
                players = ["afplay"] if sys.platform == "darwin" else ["paplay", "aplay"]
                self.cmd = next((p for p in players if shutil.which(p)), None)
                self.mode = "cmd" if self.cmd else None
        except Exception as e:
            print(f"Звук отключён: {e}")

    def play(self):
        try:
            if self.mode == "winsound":
                self._winsound.PlaySound(
                    self.path, self._winsound.SND_FILENAME | self._winsound.SND_ASYNC)
            elif self.mode == "cmd":
                subprocess.Popen([self.cmd, self.path],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
