"""
generator.py
Creates a 1-minute analog horror video (4:3, 24fps) per run.
Designed for GitHub Actions: produces an artifact (.mp4). No YouTube upload.

Fixes included:
- Guaranteed output even if scraping returns few images (progressive relax + procedural fallback images)
- Correct Ken Burns cropping to enforce 4:3 (prevents size mismatch with overlay)
- Overlays generated using actual frame dimensions (prevents broadcasting errors)
- Tracking errors / timecode / REC / dropouts / freeze handled in effects.py
- Entity flash: masked + micro-stutter, imperceptible (~0.1s)
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

from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
)

from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise
from gtts import gTTS

from .scraper import ScrapeConfig, scrape_images, fetch_normal_headline, pick_keywords
from .effects import (
    lore_string,
    make_camcorder_overlay_rgba,
    VhsProcessor,
    VhsParams,
    apply_entity_stutter,
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


# ----------------- AUDIO -----------------

def _save_tts(text: str, out_wav: str):
    """gTTS outputs mp3; convert to wav via pydub (ffmpeg required)."""
    mp3 = out_wav.replace(".wav", ".mp3")
    gTTS(text=text, lang="en", slow=False).save(mp3)
    seg = AudioSegment.from_file(mp3).set_channels(1).set_frame_rate(44100)
    seg.export(out_wav, format="wav")


def generate_drone(out_wav: str, duration_ms: int, rng: random.Random):
    """Low-frequency drone + noisy pad using pydub generators."""
    base_freq = rng.choice([38, 42, 47, 55, 62])
    overtone = base_freq * rng.choice([2, 3, 4])

    drone = Sine(base_freq).to_audio_segment(duration=duration_ms).apply_gain(-12)
    pad = Sine(overtone).to_audio_segment(duration=duration_ms).apply_gain(-22)

    # slow fake “LFO” via chunked gain
    step = 600
    chunks = []
    for i in range(0, duration_ms, step):
        g = -24 + (rng.random() * 8)
        chunks.append(pad[i:i+step].apply_gain(g))
    pad = sum(chunks)

    noise = WhiteNoise().to_audio_segment(duration=duration_ms).apply_gain(-38)
    noise = noise.low_pass_filter(1200).high_pass_filter(40)

    mix = drone.overlay(pad).overlay(noise)

    # mild distortion coloration
    if rng.random() < 0.9:
        mix = mix.apply_gain(3).low_pass_filter(rng.randint(1200, 2400)).apply_gain(-3)

    mix = mix.set_channels(1).set_frame_rate(44100)
    mix.export(out_wav, format="wav")


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
        width=76,
    )
    return narration, weird_line


# ----------------- VISUAL HELPERS -----------------

def _kenburns_clip_force_43(image_path: str, w: int, h: int, dur: float, rng: random.Random) -> ImageClip:
    """
    Ken Burns over a still image but ALWAYS output size (w,h).
    This prevents the overlay broadcast mismatch you hit.
    """
    clip = ImageClip(image_path).set_duration(dur)

    iw, ih = clip.size
    # scale to cover target + safety margin
    scale = max(w / iw, h / ih) * (1.10 + rng.random() * 0.25)
    clip = clip.resize(scale)

    cw, ch = clip.size
    max_x = max(0, cw - w)
    max_y = max(0, ch - h)

    x0 = rng.uniform(0, max_x) if max_x else 0
    y0 = rng.uniform(0, max_y) if max_y else 0
    x1 = rng.uniform(0, max_x) if max_x else 0
    y1 = rng.uniform(0, max_y) if max_y else 0

    # move inside the big scaled image
    def crop_at(t):
        a = t / dur if dur > 0 else 0.0
        x = x0 + (x1 - x0) * a
        y = y0 + (y1 - y0) * a
        return clip.crop(x1=x, y1=y, width=w, height=h)

    # MoviePy wants a clip, not a function returning clip, so:
    # we approximate by setting a position and then cropping from (0,0)
    # after putting it on a bigger canvas. The simplest reliable path:
    clip = clip.set_position(lambda t: (- (x0 + (x1-x0)*(t/dur)), - (y0 + (y1-y0)*(t/dur))))
    # Force final size deterministically
    clip = clip.on_color(size=(w, h), color=(0, 0, 0), pos=(0, 0))
    clip = clip.crop(x1=0, y1=0, width=w, height=h)
    return clip


def _entity_bgr(entity_path: str, w: int, h: int) -> np.ndarray:
    img = cv2.imread(entity_path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "?", (w // 2 - 20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6, cv2.LINE_AA)
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _merge_overlays(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Alpha-over for two RGBA numpy images of same size."""
    aa = a.astype(np.float32)
    bb = b.astype(np.float32)
    aA = aa[:, :, 3:4] / 255.0
    bA = bb[:, :, 3:4] / 255.0
    outA = bA + aA * (1 - bA)
    outRGB = (bb[:, :, :3] * bA + aa[:, :, :3] * aA * (1 - bA)) / np.clip(outA, 1e-6, 1.0)
    out = np.concatenate([outRGB, outA * 255.0], axis=2)
    return np.clip(out, 0, 255).astype(np.uint8)


def _text_overlay_rgba(w: int, h: int, lore: str, weird_line: str, rng: random.Random, t: float) -> np.ndarray:
    """Lore blocks + weird line overlay (RGBA)."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    def load_font(sz):
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size=sz)
                except Exception:
                    pass
        return ImageFont.load_default()

    font = load_font(max(14, int(h * 0.035)))
    small = load_font(max(12, int(h * 0.028)))

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

    if rng.random() < 0.22:
        for _ in range(rng.randint(6, 16)):
            cx = rng.randint(int(w * 0.07), int(w * 0.93))
            cy = rng.randint(int(h * 0.55), int(h * 0.93))
            d.text((cx, cy), rng.choice(["#", "%", "?", "∎", "░", "▒"]), font=small, fill=(255, 255, 255, 120))

    return np.array(img)


def _procedural_fallback_images(img_dir: str, rng: random.Random, count: int = 8, w: int = 960, h: int = 720) -> List[str]:
    """
    If scraping fails, generate synthetic "liminal tech" images:
    CRT glow, scanlines, noise, signage rectangles.
    """
    from PIL import Image, ImageDraw, ImageFont

    os.makedirs(img_dir, exist_ok=True)
    out_paths = []

    def load_font(sz):
        for p in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size=sz)
                except Exception:
                    pass
        return ImageFont.load_default()

    font = load_font(28)

    for i in range(count):
        img = Image.new("RGB", (w, h), (rng.randint(0, 15), rng.randint(0, 15), rng.randint(0, 15)))
        d = ImageDraw.Draw(img)

        # faint gradient
        for y in range(0, h, 3):
            g = int(10 + 35 * (y / max(1, h)) + rng.randint(-3, 3))
            d.line([(0, y), (w, y)], fill=(g, g, g))

        # random "rooms"/panels
        for _ in range(rng.randint(6, 14)):
            x0 = rng.randint(-40, w - 1)
            y0 = rng.randint(-40, h - 1)
            x1 = min(w, x0 + rng.randint(80, 420))
            y1 = min(h, y0 + rng.randint(60, 300))
            col = (rng.randint(15, 55), rng.randint(15, 55), rng.randint(15, 55))
            d.rectangle([x0, y0, x1, y1], outline=(90, 90, 90), fill=col)

        # fake signage
        if rng.random() < 0.9:
            s = rng.choice(["MAINTENANCE", "AUTHORIZED ONLY", "NO EXIT", "CAM 02", "ROOM 7B", "STAFF"])
            d.rectangle([int(w*0.08), int(h*0.12), int(w*0.45), int(h*0.18)], fill=(0, 0, 0))
            d.text((int(w*0.09), int(h*0.125)), s, font=font, fill=(220, 220, 220))

        # convert to numpy for scanlines/noise quick pass
        arr = np.array(img).astype(np.uint8)
        # scanlines
        arr[::2, :, :] = (arr[::2, :, :] * (0.78 + rng.random()*0.08)).astype(np.uint8)
        # noise
        n = np.random.normal(0, 18, (h, w, 1)).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + n, 0, 255).astype(np.uint8)

        p = os.path.join(img_dir, f"fallback_{i:02d}.jpg")
        Image.fromarray(arr).save(p, quality=92)
        out_paths.append(p)

    return out_paths


def _scrape_with_fallbacks(img_dir: str, rng: random.Random, seed: Optional[int]) -> List[str]:
    """
    Try scraping with progressively looser constraints.
    If still not enough images, generate procedural fallback images.
    """
    attempts = [
        dict(min_width=960, min_height=720, max_images=18),
        dict(min_width=720, min_height=540, max_images=22),
        dict(min_width=480, min_height=360, max_images=26),
        dict(min_width=0,   min_height=0,   max_images=30),
    ]

    best = []
    for a in attempts:
        cfg = ScrapeConfig(
            out_dir=img_dir,
            max_images=a["max_images"],
            min_width=a["min_width"],
            min_height=a["min_height"],
            seed=seed,
        )
        try:
            imgs = scrape_images(cfg)
        except Exception:
            imgs = []

        if len(imgs) > len(best):
            best = imgs

        if len(best) >= 6:
            break

    if len(best) < 3:
        best = best + _procedural_fallback_images(img_dir, rng, count=10)
    return best


# ----------------- MAIN GENERATION -----------------

def generate_video(cfg: GenConfig) -> str:
    rng = random.Random(cfg.seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    run_media = os.path.join(cfg.out_dir, "media")
    img_dir = os.path.join(run_media, "images")
    os.makedirs(img_dir, exist_ok=True)

    # 1) images (never fail)
    images = _scrape_with_fallbacks(img_dir, rng, cfg.seed)
    if len(images) < 3:
        raise RuntimeError("Still not enough images after fallbacks (unexpected).")

    # 2) weird element + script
    headline = fetch_normal_headline(cfg.seed)
    keywords = pick_keywords(cfg.seed)
    narration, weird_line = build_script_text(rng, headline, keywords)

    # 3) audio
    tts_wav = os.path.join(run_media, "tts.wav")
    drone_wav = os.path.join(run_media, "drone.wav")
    mix_wav = os.path.join(run_media, "mix.wav")

    _save_tts(narration, tts_wav)
    generate_drone(drone_wav, int(cfg.duration_s * 1000), rng)

    tts = AudioSegment.from_file(tts_wav)
    drone = AudioSegment.from_file(drone_wav)

    # warble
    warbled = AudioSegment.silent(duration=len(drone))
    step = 480
    for i in range(0, len(drone), step):
        g = -10 + (rng.random() * 5)
        warbled += drone[i:i+step].apply_gain(g)
    drone = warbled

    mix = drone.overlay(tts.apply_gain(-1))
    mix = mix.low_pass_filter(rng.randint(1800, 3200)).apply_gain(-1)
    mix.export(mix_wav, format="wav")

    # 4) visuals base
    seg_count = rng.randint(7, 11)
    seg_dur = cfg.duration_s / seg_count
    chosen = [rng.choice(images) for _ in range(seg_count)]

    clips = [
        _kenburns_clip_force_43(p, cfg.width, cfg.height, seg_dur, rng)
        for p in chosen
    ]
    base_clip = concatenate_videoclips(clips, method="compose").set_duration(cfg.duration_s)
    # Double-force size just in case (paranoia to avoid mismatches)
    base_clip = base_clip.resize((cfg.width, cfg.height))

    # 5) entity scheduling (0.1s ≈ 2-3 frames at 24fps)
    entity_img = _entity_bgr(rng.choice(images), cfg.width, cfg.height)
    entity_time = rng.uniform(4.0, cfg.duration_s - 4.0)
    entity_frames = max(2, int(0.10 * cfg.fps))
    entity_times = [entity_time + (i / cfg.fps) for i in range(entity_frames)]
    if rng.random() < 0.85:
        entity_times.insert(1, entity_times[0])  # micro-stutter duplicate frame

    lore = lore_string(rng)
    lore_start = rng.uniform(8, 16)
    lore_end = min(cfg.duration_s, lore_start + rng.uniform(16, 28))

    # 6) VHS processor (heavy)
    processor = VhsProcessor(rng, VhsParams(
        grain=0.95,
        scanlines=0.90,
        chroma_shift=rng.randint(1, 3),
        vignette=0.35,
        tracking_strength=0.95,
        dropout_prob=0.017,
        freeze_prob=0.014,
    ))

    def make_frame(t: float):
        rgb = base_clip.get_frame(t)  # RGB np.uint8
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # IMPORTANT: derive overlay size from actual frame
        fh, fw = bgr.shape[:2]

        overlay = make_camcorder_overlay_rgba(fw, fh, t, rng, show_tracking=True)

        if lore_start <= t <= lore_end:
            overlay2 = _text_overlay_rgba(fw, fh, lore, weird_line, rng, t)
            overlay = _merge_overlays(overlay, overlay2)

        # entity flash: masked blend on a few frames (imperceptible)
        for tt in entity_times:
            if abs(t - tt) < (0.5 / cfg.fps):
                bgr = apply_entity_stutter(bgr, entity_img, rng, opacity=0.12 + rng.random() * 0.06)
                break

        out = processor.process(bgr, t, overlay_rgba=overlay)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    v = VideoClip(make_frame, duration=cfg.duration_s).set_fps(cfg.fps)
    v = v.set_audio(AudioFileClip(mix_wav).set_duration(cfg.duration_s))

    os.makedirs(os.path.dirname(cfg.out_mp4) or ".", exist_ok=True)
    v.write_videofile(
        cfg.out_mp4,
        fps=cfg.fps,
        codec="libx264",
        audio_codec="aac",
        bitrate="3500k",
        threads=2,
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
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

    cfg = GenConfig(
        out_mp4=args.out,
        out_dir=out_dir,
        duration_s=float(args.duration),
        seed=seed,
    )
    path = generate_video(cfg)
    print(f"OK: generated {path} (seed={seed})")


if __name__ == "__main__":
    main()


