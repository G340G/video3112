"""
effects.py
OpenCV VHS/analog effects + overlays.

Added for your request:
- tracking errors: horizontal bands that slip (bande orizzontali che slittano)
- timecode/tape overlay camcorder style (REC, SP, TRACKING)
- freeze + random black/green dropouts
- entity frame with random mask + stutter (imperceptible but unsettling)
"""
from __future__ import annotations

import base64
import math
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# ---------- Lore text ----------

def caesar(s: str, shift: int) -> str:
    out = []
    for ch in s:
        o = ord(ch)
        if 97 <= o <= 122:
            out.append(chr(((o - 97 + shift) % 26) + 97))
        elif 65 <= o <= 90:
            out.append(chr(((o - 65 + shift) % 26) + 65))
        else:
            out.append(ch)
    return "".join(out)

def lore_string(rng: random.Random) -> str:
    base = rng.choice([
        "THE CAMERA REMEMBERS",
        "DON'T LOOK AT THE CORNERS",
        "SIGNAL IS CLEAN [REDACTED]",
        "THE HALLWAY HAS AN ENDLESS LOOP",
        "WE LOST THE TAPES IN 1997",
        "SUBJECT REFUSES TO WAKE",
        "EVERYTHING IS FINE UNTIL IT ISN'T",
        "THE WEATHER IS FINE. [REDACTED] IS WATCHING.",
        "IF YOU HEAR IT, STOP RECORDING"
    ])
    shift = rng.randint(1, 25)
    c = caesar(base, shift)
    b = base64.b64encode(c.encode("utf-8")).decode("ascii")
    chunk = rng.randint(10, 18)
    blocks = " ".join([b[i:i+chunk] for i in range(0, len(b), chunk)])
    return blocks


# ---------- Overlay (camcorder style) ----------

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def make_camcorder_overlay_rgba(w: int, h: int, t_seconds: float, rng: random.Random, show_tracking: bool = True) -> np.ndarray:
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    font = _load_font(max(14, int(h * 0.04)))

    rec_on = (int(t_seconds * 2) % 2) == 0
    if rec_on:
        d.rectangle([int(w*0.06), int(h*0.07), int(w*0.06)+16, int(h*0.07)+16], fill=(220, 30, 30, 255))
        d.text((int(w*0.06)+22, int(h*0.064)), "REC", font=font, fill=(255, 255, 255, 240))

    d.text((int(w*0.78), int(h*0.064)), rng.choice(["SP", "LP", "EP"]), font=font, fill=(255, 255, 255, 220))
    d.text((int(w*0.06), int(h*0.90)), rng.choice(["CAM", "PLAY", "VTR"]), font=font, fill=(255, 255, 255, 200))

    fps = 24
    total_frames = int(t_seconds * fps)
    hh = total_frames // (fps*3600)
    mm = (total_frames // (fps*60)) % 60
    ss = (total_frames // fps) % 60
    ff = total_frames % fps
    tc = f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"
    d.text((int(w*0.66), int(h*0.90)), tc, font=font, fill=(255, 255, 255, 220))

    if show_tracking and rng.random() < 0.96:
        tr_x = int(w*0.06) + rng.randint(-2, 2)
        tr_y = int(h*0.12) + rng.randint(-2, 2)
        d.text((tr_x, tr_y), "TRACKING", font=font, fill=(255, 255, 255, 170))

    return np.array(img)


def alpha_blend_bgr_with_rgba(frame_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    bgr = frame_bgr.astype(np.float32)
    rgba = overlay_rgba.astype(np.float32)
    a = rgba[:, :, 3:4] / 255.0
    rgb = rgba[:, :, :3][:, :, ::-1]
    out = bgr * (1 - a) + rgb * a
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------- VHS effects ----------

@dataclass
class VhsParams:
    grain: float = 0.75
    scanlines: float = 0.75
    chroma_shift: int = 2
    vignette: float = 0.30
    tracking_strength: float = 0.9
    dropout_prob: float = 0.012
    freeze_prob: float = 0.010


def add_grain(frame: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return frame
    h, w = frame.shape[:2]
    n = np.random.normal(0, 18 + 30*amount, (h, w, 1)).astype(np.float32)
    out = frame.astype(np.float32) + n
    return np.clip(out, 0, 255).astype(np.uint8)


def add_scanlines(frame: np.ndarray, amount: float, rng: random.Random) -> np.ndarray:
    if amount <= 0:
        return frame
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)
    line = np.arange(h).reshape(h, 1)
    mask = ((line % 2) == 0).astype(np.float32)
    strength = (0.10 + 0.25 * amount) * (0.75 + 0.5 * rng.random())
    out *= (1 - strength * mask[:, :, None])
    if rng.random() < 0.25 * amount:
        y0 = rng.randint(0, h-1)
        band_h = rng.randint(6, 18)
        out[y0:y0+band_h, :, :] *= (0.55 + 0.2 * rng.random())
    return np.clip(out, 0, 255).astype(np.uint8)


def chromatic_aberration(frame: np.ndarray, shift: int, rng: random.Random) -> np.ndarray:
    if shift <= 0:
        return frame
    b, g, r = cv2.split(frame)
    dx = shift + rng.randint(-1, 1)
    dy = rng.randint(-1, 1)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    r2 = cv2.warpAffine(r, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
    M2 = np.float32([[1, 0, -dx], [0, 1, -dy]])
    b2 = cv2.warpAffine(b, M2, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b2, g, r2])


def vignette(frame: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return frame
    h, w = frame.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h/2.0, w/2.0
    d = ((x - cx)**2 + (y - cy)**2) / (max(h, w)**2)
    v = 1 - amount * (d * 3.2)
    v = np.clip(v, 0.55, 1.0).astype(np.float32)
    out = frame.astype(np.float32) * v[:, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def tracking_errors(frame: np.ndarray, strength: float, t: float, rng: random.Random) -> np.ndarray:
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    out = frame.copy()
    bands = int(1 + strength * 4)
    for _ in range(bands):
        band_h = rng.randint(int(h*0.02), int(h*0.10))
        y0 = rng.randint(0, max(0, h - band_h - 1))
        base = math.sin(t * (1.7 + rng.random()*2.2) + rng.random()*6.28)
        off = int(base * (10 + 60*strength) + rng.randint(-6, 6))
        for y in range(y0, y0 + band_h):
            row_off = int(off * (0.4 + 0.6 * (y - y0) / max(1, band_h-1)))
            out[y, :, :] = np.roll(out[y, :, :], row_off, axis=0)
        if rng.random() < 0.55:
            ly = min(h-1, y0 + rng.randint(0, band_h-1))
            out[ly:ly+1, :, :] = np.clip(out[ly:ly+1, :, :].astype(np.int16) + 70, 0, 255).astype(np.uint8)
    return out


def dropouts(frame: np.ndarray, rng: random.Random, prob: float) -> np.ndarray:
    if rng.random() > prob:
        return frame
    h, w = frame.shape[:2]
    out = frame.copy()
    green = rng.random() < 0.55
    color = (0, 255, 0) if green else (0, 0, 0)
    for _ in range(rng.randint(1, 3)):
        x0 = rng.randint(0, w-1)
        y0 = rng.randint(0, h-1)
        ww = rng.randint(int(w*0.05), int(w*0.35))
        hh = rng.randint(int(h*0.02), int(h*0.20))
        x1 = min(w, x0 + ww)
        y1 = min(h, y0 + hh)
        out[y0:y1, x0:x1, :] = color
    if rng.random() < 0.25:
        y0 = rng.randint(0, h-1)
        hh = rng.randint(2, 10)
        out[y0:y0+hh, :, :] = color
    return out


def random_mask(w: int, h: int, rng: random.Random) -> np.ndarray:
    img = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(4, 9)):
        x = rng.randint(-int(w*0.1), int(w*1.1))
        y = rng.randint(-int(h*0.1), int(h*1.1))
        rx = rng.randint(int(w*0.04), int(w*0.18))
        ry = rng.randint(int(h*0.04), int(h*0.22))
        d.ellipse([x-rx, y-ry, x+rx, y+ry], fill=rng.randint(120, 255))
    arr = np.array(img).astype(np.float32)
    k = rng.choice([5, 7, 9])
    arr = cv2.GaussianBlur(arr, (k, k), 0)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def apply_entity_stutter(frame: np.ndarray, entity_bgr: np.ndarray, rng: random.Random, opacity: float = 0.14) -> np.ndarray:
    h, w = frame.shape[:2]
    ent = cv2.resize(entity_bgr, (w, h), interpolation=cv2.INTER_AREA)
    jx, jy = rng.randint(-2, 2), rng.randint(-2, 2)
    M = np.float32([[1, 0, jx], [0, 1, jy]])
    ent = cv2.warpAffine(ent, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    mask = random_mask(w, h, rng).astype(np.float32) / 255.0
    mask = mask[:, :, None]
    out = frame.astype(np.float32)
    blended = out * (1 - opacity*mask) + ent.astype(np.float32) * (opacity*mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


class VhsProcessor:
    def __init__(self, rng: random.Random, params: Optional[VhsParams] = None):
        self.rng = rng
        self.params = params or VhsParams()
        self._freeze_frames_left = 0
        self._frozen: Optional[np.ndarray] = None

    def process(self, frame_bgr: np.ndarray, t: float, overlay_rgba: Optional[np.ndarray] = None) -> np.ndarray:
        if self._freeze_frames_left > 0 and self._frozen is not None:
            self._freeze_frames_left -= 1
            base = self._frozen.copy()
        else:
            base = frame_bgr
            if self.rng.random() < self.params.freeze_prob:
                self._frozen = frame_bgr.copy()
                self._freeze_frames_left = self.rng.randint(2, 10)

        out = chromatic_aberration(base, self.params.chroma_shift, self.rng)
        out = tracking_errors(out, self.params.tracking_strength, t, self.rng)
        out = add_scanlines(out, self.params.scanlines, self.rng)
        out = add_grain(out, self.params.grain)
        out = vignette(out, self.params.vignette)
        out = dropouts(out, self.rng, self.params.dropout_prob)
        if overlay_rgba is not None:
            out = alpha_blend_bgr_with_rgba(out, overlay_rgba)
        return out
