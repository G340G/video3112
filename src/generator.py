"""
generator.py
Creates a 1-minute analog horror video (4:3, 24fps) per run.
No YouTube upload: GitHub Actions produces an artifact (.mp4).

Fixes:
- Robust scraping with retries + relaxed min size if needed
- Guaranteed fallback by generating synthetic "liminal/VHS" images when scraping is insufficient

Run:
python -m src.generator --out out/analog_horror.mp4 --seed 1234 --duration 60
"""
from __future__ import annotations

import argparse
import os
import random
import textwrap
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips
)

from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from gtts import gTTS

from .scraper import ScrapeConfig, scrape_images, fetch_normal_headline, pick_keywords
from .effects import (
    lore_string, make_camcorder_overlay_rgba, VhsProcessor, VhsParams, apply_entity_stutter
)


@dataclass
class GenConfig:
    out_mp4: str
    out_dir: str
    duration_s: float = 60.0
    fps: int = 24
    width: int = 960   # 4:3
    height: int = 720  # 4:3
    seed: Optional[int] = None


# ---------------------------
# Audio
# ---------------------------

def _save_tts(text: str, out_wav: str):
    """
    gTTS outputs mp3; convert to wav via pydub (ffmpeg required).
    """
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)
    mp3 = out_wav.replace(".wav", ".mp3")
    gTTS(text=text, lang="en", slow=False).save(mp3)
    seg = AudioSegment.from_file(mp3).set_channels(1).set_frame_rate(44100)
    seg.export(out_wav, format="wav")


def generate_drone(out_wav: str, duration_ms: int, rng: random.Random):
    """
    Low-frequency drone + noisy pad using pydub generators.
    """
    os.makedirs(os.path.dirname(out_wav) or ".", exist_ok=True)

    base_freq = rng.choice([38, 42, 47, 55, 62])
    overtone = base_freq * rng.choice([2, 3, 4])

    drone = Sine(base_freq).to_audio_segment(duration=duration_ms).apply_gain(-12)
    pad = Sine(overtone).to_audio_segment(duration=duration_ms).apply_gain(-22)

    # Fake slow LFO by chunked gain variation
    chunks = []
    step = 600
    for i in range(0, duration_ms, step):
        g = -24 + (rng.random() * 8)
        chunks.append(pad[i:i+step].apply_gain(g))
    pad = sum(chunks)

    noise = WhiteNoise().to_audio_segment(duration=duration_ms).apply_gain(-38)
    noise = noise.low_pass_filter(1200).high_pass_filter(40)

    mix = drone.overlay(pad).overlay(noise)

    # Mild distortion coloration
    if rng.random() < 0.9:
        mix = mix.apply_gain(3).low_pass_filter(rng.randint(1200, 2400)).apply_gain(-3)

    mix = mix.set_channels(1).set_frame_rate(44100)
    mix.export(out_wav, format="wav")


# ---------------------------
# Text / script
# ---------------------------

def build_script_text(rng: random.Random, headline: str, keywords: List[str]) -> Tuple[str, str]:
    eerie_lines = [
        "The weather is fine.",
        "Nothing is wrong in the corridor.",
        "Your device is working as intended.",
        "Please remain still while the signal stabilizes.",
        "Do not rewind.",
        "If you see the figure, do not acknowledge it.",
    ]
    redacted = "[REDACTED]" * rng.randint(1, 3)
    weird_line = f"{rng.choice(eerie_lines)} {redacted} IS WATCHING."
    narration = textwrap.fill(
        f"Headline: {headline}. "
        f"Keywords: {', '.join(keywords)}. "
        f"{weird_line} "
        f"End of report.",
        width=76
    )
    return narration, weird_line


# ---------------------------
# Visual helpers
# ---------------------------

def _kenburns_clip(image_path: str, w: int, h: int, dur: float, rng: random.Random) -> ImageClip:
    """
    Ken Burns style: slow zoom + pan.
    """
    clip = ImageClip(image_path).set_duration(dur)
    iw, ih = clip.size

    # Make sure it covers frame
    scale = max(w / iw, h / ih) * (1.05 + rng.random() * 0.22)
    clip = clip.resize(scale)

    cw, ch = clip.size
    max_x = max(0, cw - w)
    max_y = max(0, ch - h)

    x0 = rng.uniform(0, max_x) if max_x else 0
    y0 = rng.uniform(0, max_y) if max_y else 0
    x1 = rng.uniform(0, max_x) if max_x else 0
    y1 = rng.uniform(0, max_y) if max_y else 0

    def pos(t):
        a = t / dur if dur > 0 else 0
        x = x0 + (x1 - x0) * a
        y = y0 + (y1 - y0) * a
        return (-x, -y)

    return clip.set_position(pos).crop(x1=w, y1=h)


def _entity_bgr(entity_path: str, w: int, h: int) -> np.ndarray:
    img = cv2.imread(entity_path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "?", (w // 2 - 20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _merge_overlays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    RGBA alpha-over: b over a
    """
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    aA = aa[:, :, 3:4] / 255.0
    bA = bb[:, :, 3:4] / 255.0
    outA = bA + aA * (1 - bA)
    outRGB = (bb[:, :, :3] * bA + aa[:, :, :3] * aA * (1 - bA)) / np.clip(outA, 1e-6, 1.0)
    out = np.concatenate([outRGB, outA * 255.0], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _text_overlay_rgba(w: int, h: int, lore: str, weird_line: str, rng: random.Random, t: float) -> np.ndarray:
    """
    Render lore blocks + weird element as translucent overlay.
    """
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    font = _load_font(max(14, int(h * 0.035)))
    small = _load_font(max(12, int(h * 0.028)))

    jx = rng.randint(-2, 2)
    jy = rng.randint(-2, 2)

    x = int(w * 0.08) + jx
    y = int(h * 0.58) + jy + int(math.sin(t * 6.0) * 2)
    box_w = int(w * 0.84)

    d.rectangle([x - 8, y - 10, x + box_w, y + int(h * 0.14)], fill=(0, 0, 0, 110))
    d.text((x, y), lore[:220], font=small, fill=(255, 255, 255, 190))

    y2 = int(h * 0.82) + rng.randint(-1, 1)
    d.rectangle([int(w * 0.06), y2 - 6, int(w * 0.94), y2 + int(h * 0.08)], fill=(0, 0, 0, 120))
    d.text((int(w * 0.08), y2), weird_line[:80], font=font, fill=(255, 255, 255, 210))

    # occasional corrupted glyphs
    if rng.random() < 0.22:
        for _ in range(rng.randint(6, 16)):
            cx = rng.randint(int(w * 0.07), int(w * 0.93))
            cy = rng.randint(int(h * 0.55), int(h * 0.93))
            d.text((cx, cy), rng.choice(["#", "%", "?", "∎", "░", "▒"]), font=small, fill=(255, 255, 255, 120))

    return np.array(img)


# ---------------------------
# Robust scraping + fallback
# ---------------------------

def _synthetic_liminal_image(w: int, h: int, rng: random.Random) -> np.ndarray:
    """
    Procedural 'liminal' still: foggy gradients, corridor-like perspective lines,
    scanlines-ish noise. Returns BGR.
    """
    # base gradient
    y = np.linspace(0, 1, h).reshape(h, 1, 1).astype(np.float32)
    x = np.linspace(0, 1, w).reshape(1, w, 1).astype(np.float32)

    base = (0.10 + 0.25 * (1 - y) + 0.15 * x).astype(np.float32)
    img = np.repeat(base, 3, axis=2)
    img *= rng.uniform(160, 220) / 255.0

    # add fog
    fog = cv2.GaussianBlur(np.random.normal(0, 1, (h, w)).astype(np.float32), (0, 0), sigmaX=rng.uniform(8, 18))
    fog = (fog - fog.min()) / (fog.max() - fog.min() + 1e-6)
    fog = fog * rng.uniform(0.18, 0.32)
    img[:, :, 0] += fog
    img[:, :, 1] += fog
    img[:, :, 2] += fog

    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # corridor perspective lines
    cx = w // 2 + rng.randint(-40, 40)
    floor_y = int(h * rng.uniform(0.62, 0.76))
    for k in range(rng.randint(10, 18)):
        x1 = cx + int((k - 9) * rng.uniform(16, 32))
        cv2.line(out, (cx, floor_y), (x1, h), (rng.randint(80, 130),) * 3, 1, cv2.LINE_AA)

    # ceiling/floor bands
    cv2.rectangle(out, (0, 0), (w, int(h * 0.12)), (rng.randint(10, 30),) * 3, -1)
    cv2.rectangle(out, (0, int(h * 0.85)), (w, h), (rng.randint(10, 30),) * 3, -1)

    # mild vhs-ish noise lines
    for _ in range(rng.randint(3, 9)):
        yy = rng.randint(0, h - 1)
        out[yy:yy+1, :, :] = np.clip(out[yy:yy+1, :, :].astype(np.int16) + rng.randint(-30, 40), 0, 255).astype(np.uint8)

    # subtle text ghost
    if rng.random() < 0.5:
        pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        f = _load_font(max(16, int(h * 0.04)))
        msg = rng.choice(["EVIDENCE", "ARCHIVE", "PLAY", "NO SIGNAL", "TRACKING", "ROOM"])
        d.text((rng.randint(20, w - 180), rng.randint(20, h - 80)), msg, font=f, fill=(255, 255, 255, rng.randint(20, 60)))
        out = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    return out


def _generate_synthetic_images(img_dir: str, n: int, w: int, h: int, rng: random.Random) -> List[str]:
    os.makedirs(img_dir, exist_ok=True)
    paths = []
    for i in range(n):
        bgr = _synthetic_liminal_image(w, h, rng)
        p = os.path.join(img_dir, f"synthetic_{i:03d}.jpg")
        cv2.imwrite(p, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        paths.append(p)
    return paths


def _scrape_with_retries(img_dir: str, seed: int, wanted: int, rng: random.Random) -> List[str]:
    """
    Try multiple scrape attempts:
    - vary seed a bit
    - relax min size if too strict
    - aggregate unique files
    """
    all_imgs: List[str] = []
    seen = set()

    # progressively relax min sizes to avoid empty runs
    attempts = [
        (960, 720),
        (800, 600),
        (640, 480),
    ]

    for pass_idx, (minw, minh) in enumerate(attempts):
        for k in range(3):  # 3 tries per pass
            scfg = ScrapeConfig(
                out_dir=img_dir,
                max_images=max(wanted * 2, 18),
                min_width=minw,
                min_height=minh,
                seed=seed + pass_idx * 100 + k * 17,
            )
            try:
                imgs = scrape_images(scfg)
            except Exception:
                imgs = []

            for p in imgs:
                if p not in seen:
                    seen.add(p)
                    all_imgs.append(p)

            if len(all_imgs) >= wanted:
                return all_imgs

    return all_imgs


# ---------------------------
# Main video generation
# ---------------------------

def generate_video(cfg: GenConfig) -> str:
    rng = random.Random(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    run_media = os.path.join(cfg.out_dir, "media")
    img_dir = os.path.join(run_media, "images")
    os.makedirs(img_dir, exist_ok=True)

    # Robust scrape
    images = _scrape_with_retries(img_dir=img_dir, seed=int(cfg.seed or 0), wanted=10, rng=rng)

    # Fallback: synth images (guaranteed)
    if len(images) < 3:
        synth = _generate_synthetic_images(img_dir=img_dir, n=12, w=cfg.width, h=cfg.height, rng=rng)
        # merge, keep unique
        seen = set(images)
        for p in synth:
            if p not in seen:
                images.append(p)
                seen.add(p)

    # Still guard (should never happen)
    if len(images) < 3:
        raise RuntimeError("Image pipeline failed even after synthetic fallback.")

    # Weird element + lore
    headline = fetch_normal_headline(cfg.seed)
    keywords = pick_keywords(cfg.seed)
    narration, weird_line = build_script_text(rng, headline, keywords)

    # Audio
    tts_wav = os.path.join(run_media, "tts.wav")
    drone_wav = os.path.join(run_media, "drone.wav")
    mix_wav = os.path.join(run_media, "mix.wav")

    _save_tts(narration, tts_wav)
    generate_drone(drone_wav, int(cfg.duration_s * 1000), rng)

    tts = AudioSegment.from_file(tts_wav)
    drone = AudioSegment.from_file(drone_wav)

    # subtle warble
    warbled = AudioSegment.silent(duration=len(drone))
    step = 480
    for i in range(0, len(drone), step):
        g = -10 + (rng.random() * 5)
        warbled += drone[i:i+step].apply_gain(g)
    drone = warbled

    mix = drone.overlay(tts.apply_gain(-1))
    mix = mix.low_pass_filter(rng.randint(1800, 3200)).apply_gain(-1)
    mix.export(mix_wav, format="wav")

    # Base visuals from images
    seg_count = rng.randint(7, 11)
    seg_dur = cfg.duration_s / seg_count
    chosen = [rng.choice(images) for _ in range(seg_count)]
    clips = [_kenburns_clip(p, cfg.width, cfg.height, seg_dur, rng) for p in chosen]
    base_clip = concatenate_videoclips(clips, method="compose").set_duration(cfg.duration_s)

    # Entity: imperceptible flash (0.1s) with stutter
    entity_path = rng.choice(images)
    entity_img = _entity_bgr(entity_path, cfg.width, cfg.height)
    entity_time = rng.uniform(4.0, cfg.duration_s - 4.0)
    entity_frames = max(2, int(0.10 * cfg.fps))
    entity_times = [entity_time + (i / cfg.fps) for i in range(entity_frames)]
    if rng.random() < 0.85:
        entity_times.insert(1, entity_times[0])  # stutter duplicate

    # Lore schedule
    lore = lore_string(rng)
    lore_start = rng.uniform(8, 16)
    lore_end = min(cfg.duration_s, lore_start + rng.uniform(16, 28))

    # VHS processor (tracking/freeze/dropouts heavy)
    processor = VhsProcessor(
        rng,
        VhsParams(
            grain=0.95,
            scanlines=0.90,
            chroma_shift=rng.randint(1, 3),
            vignette=0.35,
            tracking_strength=0.95,  # bande orizzontali che slittano
            dropout_prob=0.017,      # buchi neri/verde
            freeze_prob=0.014        # freeze casuali
        ),
    )

    def make_frame(t: float):
        rgb = base_clip.get_frame(t)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # camcorder overlay: REC / SP / TRACKING / timecode
        overlay = make_camcorder_overlay_rgba(cfg.width, cfg.height, t, rng, show_tracking=True)

        # lore overlay windows
        if lore_start <= t <= lore_end:
            overlay2 = _text_overlay_rgba(cfg.width, cfg.height, lore, weird_line, rng, t)
            overlay = _merge_overlays(overlay, overlay2)

        # entity masked flash (imperceptible)
        for tt in entity_times:
            if abs(t - tt) < (0.5 / cfg.fps):
                bgr = apply_entity_stutter(bgr, entity_img, rng, opacity=0.12 + rng.random() * 0.06)
                break

        out = processor.process(bgr, t, overlay_rgba=overlay)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    v = VideoClip(make_frame, duration=cfg.duration_s).set_fps(cfg.fps)
    v = v.set_audio(AudioFileClip(mix_wav).set_duration(cfg.duration_s))

    # Export
    os.makedirs(os.path.dirname(cfg.out_mp4) or ".", exist_ok=True)
    v.write_videofile(
        cfg.out_mp4,
        fps=cfg.fps,
        codec="libx264",
        audio_codec="aac",
        bitrate="3500k",
        threads=2,
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p"]
    )
    return cfg.out_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/analog_horror.mp4")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "big")
    out_dir = os.path.dirname(args.out) or "out"

    cfg = GenConfig(out_mp4=args.out, out_dir=out_dir, duration_s=float(args.duration), seed=seed)
    path = generate_video(cfg)
    print(f"OK: generated {path} (seed={seed})")


if __name__ == "__main__":
    main()

