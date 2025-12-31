# effects.py
from __future__ import annotations

import base64
import math
import os
import random
from typing import Tuple

import cv2
import numpy as np


def caesar_cipher(text: str, shift: int) -> str:
    out = []
    for ch in text:
        if "a" <= ch <= "z":
            out.append(chr((ord(ch) - 97 + shift) % 26 + 97))
        elif "A" <= ch <= "Z":
            out.append(chr((ord(ch) - 65 + shift) % 26 + 65))
        else:
            out.append(ch)
    return "".join(out)


def lore_string(seed: int) -> str:
    rnd = random.Random(seed)
    phrases = [
        "DO NOT TRUST THE SIGNAL",
        "THE HALLWAY REPEATS",
        "THE TAPES REMEMBER YOU",
        "EYES BEHIND THE GLASS",
        "THE WEATHER IS FINE",
        "PLEASE REMAIN CALM",
        "YOU WERE NEVER HERE",
        "THE CAMERA BLINKED FIRST",
    ]
    raw = f"{rnd.choice(phrases)} :: {rnd.randint(1000,9999)} :: {rnd.choice(['NUL','CRC','VX','HISS'])}"
    shifted = caesar_cipher(raw, rnd.randint(3, 19))
    b64 = base64.b64encode(shifted.encode("utf-8")).decode("ascii")
    # shorten a bit for overlay
    return b64[: rnd.randint(28, 46)]


def add_grain(frame_bgr: np.ndarray, amount: float, seed: int) -> np.ndarray:
    rnd = np.random.default_rng(seed)
    h, w = frame_bgr.shape[:2]
    noise = rnd.normal(0, 18.0 * amount, (h, w, 1)).astype(np.float32)
    out = frame_bgr.astype(np.float32)
    out += noise
    return np.clip(out, 0, 255).astype(np.uint8)


def add_scanlines(frame_bgr: np.ndarray, strength: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.astype(np.float32)
    # darken every other line
    line = (np.arange(h) % 2).astype(np.float32)[:, None]
    factor = 1.0 - strength * 0.25 * line
    out *= factor[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def chromatic_aberration(frame_bgr: np.ndarray, shift_px: int) -> np.ndarray:
    b, g, r = cv2.split(frame_bgr)
    h, w = b.shape

    def shift_channel(ch: np.ndarray, dx: int, dy: int) -> np.ndarray:
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(ch, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # shift red and blue opposite directions
    r2 = shift_channel(r, shift_px, 0)
    b2 = shift_channel(b, -shift_px, 0)
    return cv2.merge([b2, g, r2])


def vhs_jitter(frame_bgr: np.ndarray, seed: int, max_shift: int = 8) -> np.ndarray:
    rnd = random.Random(seed)
    h, w = frame_bgr.shape[:2]
    dx = rnd.randint(-max_shift, max_shift)
    dy = rnd.randint(-2, 2)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def vignette(frame_bgr: np.ndarray, strength: float = 0.55) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    r = np.sqrt(x * x + y * y)
    mask = 1.0 - strength * np.clip(r, 0, 1.0)
    out = frame_bgr.astype(np.float32) * mask[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_vhs(frame_rgb: np.ndarray, t: float, seed: int) -> np.ndarray:
    """
    MoviePy gives RGB; we do processing in BGR and return RGB.
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # time-varying parameters
    shift = int(1 + (abs(math.sin(t * 1.7)) * 3))
    grain_amt = 0.85
    scan_strength = 0.85

    out = chromatic_aberration(frame_bgr, shift_px=shift)
    out = vhs_jitter(out, seed=seed + int(t * 1000), max_shift=6)
    out = add_scanlines(out, strength=scan_strength)
    out = add_grain(out, amount=grain_amt, seed=seed + 1337 + int(t * 10))
    out = vignette(out, strength=0.65)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def overlay_entity(base_rgb: np.ndarray, entity_rgb: np.ndarray, opacity: float) -> np.ndarray:
    h, w = base_rgb.shape[:2]
    ent = cv2.resize(entity_rgb, (w, h), interpolation=cv2.INTER_AREA)
    out = base_rgb.astype(np.float32)
    out = out * (1.0 - opacity) + ent.astype(np.float32) * opacity
    return np.clip(out, 0, 255).astype(np.uint8)
