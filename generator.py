# generator.py
from __future__ import annotations

import argparse
import os
import random
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests
from PIL import Image

from moviepy.editor import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
    VideoClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioArrayClip

from pydub import AudioSegment
from pydub.generators import Sine, WhiteNoise

import scraper
import effects

UA = "AnalogHorrorBot/1.0 (GitHub Actions)"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def download_file(url: str, out_path: Path, timeout: int = 40) -> bool:
    try:
        r = requests.get(url, headers={"User-Agent": UA}, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 10_000
    except Exception:
        return False


def load_image_as_rgb(path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    # force 4:3 fill (crop center)
    w, h = im.size
    tw, th = target_size
    target_ratio = tw / th
    src_ratio = w / h

    if src_ratio > target_ratio:
        # too wide: crop width
        new_w = int(h * target_ratio)
        x0 = (w - new_w) // 2
        im = im.crop((x0, 0, x0 + new_w, h))
    else:
        # too tall: crop height
        new_h = int(w / target_ratio)
        y0 = (h - new_h) // 2
        im = im.crop((0, y0, w, y0 + new_h))

    im = im.resize((tw, th), Image.Resampling.LANCZOS)
    return np.array(im)


def make_kenburns_clip(img_rgb: np.ndarray, dur: float, fps: int, seed: int) -> VideoClip:
    rnd = random.Random(seed)
    h, w = img_rgb.shape[:2]

    base = ImageClip(img_rgb).set_duration(dur)

    # slow zoom + small pan
    z0 = rnd.uniform(1.00, 1.05)
    z1 = rnd.uniform(1.08, 1.18)
    x0 = rnd.uniform(-0.02, 0.02)
    y0 = rnd.uniform(-0.02, 0.02)
    x1 = rnd.uniform(-0.03, 0.03)
    y1 = rnd.uniform(-0.03, 0.03)

    def pos_at(t):
        a = t / max(dur, 1e-6)
        x = (x0 * (1 - a) + x1 * a) * w
        y = (y0 * (1 - a) + y1 * a) * h
        return (x, y)

    def scale_at(t):
        a = t / max(dur, 1e-6)
        return z0 * (1 - a) + z1 * a

    clip = base.resize(lambda t: scale_at(t)).set_position(lambda t: pos_at(t))
    # After scaling/panning, crop back to full frame
    clip = clip.crop(x_center=w // 2, y_center=h // 2, width=w, height=h)
    clip = clip.set_fps(fps)
    return clip


def vhs_transform(clip: VideoClip, seed: int) -> VideoClip:
    return clip.fl_image(lambda fr: effects.apply_vhs(fr, t=0.0, seed=seed)).fl(
        lambda gf, t: effects.apply_vhs(gf(t), t=t, seed=seed)
    )


def make_lore_overlays(
    w: int,
    h: int,
    duration: float,
    headline: str,
    seed: int,
) -> List[VideoClip]:
    rnd = random.Random(seed)
    overlays: List[VideoClip] = []

    # headline (normal) + eerie suffix
    eerie = rnd.choice(
        [
            " [REDACTED] IS WATCHING.",
            " THE SIGNAL STAYS ON.",
            " DO NOT LOOK AWAY.",
            " REPORT ANY SMELL OF OZONE.",
            " THIS IS A TEST. THIS IS NOT A TEST.",
        ]
    )
    weird_line = (headline.strip() + "." + eerie).replace("..", ".")

    # bottom crawl
    bottom = (
        TextClip(
            weird_line,
            fontsize=24,
            font="DejaVu-Sans",
            color="white",
            method="caption",
            size=(w - 40, None),
        )
        .set_position(("center", h - 60))
        .set_duration(duration)
        .set_opacity(0.85)
    )
    overlays.append(bottom)

    # scattered lore strings
    for _ in range(rnd.randint(8, 14)):
        t0 = rnd.uniform(0.0, max(0.0, duration - 2.0))
        dur = rnd.uniform(0.7, 2.2)
        lore = effects.lore_string(seed + int(t0 * 1000) + rnd.randint(0, 9999))

        x = rnd.randint(20, w - 240)
        y = rnd.randint(20, h - 80)

        tc = (
            TextClip(
                lore,
                fontsize=rnd.randint(18, 28),
                font="DejaVu-Sans-Mono",
                color="white",
                method="label",
            )
            .set_position((x, y))
            .set_start(t0)
            .set_duration(dur)
            .set_opacity(rnd.uniform(0.35, 0.75))
        )
        overlays.append(tc)

    return overlays


def generate_drone_audio(duration_s: float, seed: int, out_wav: Path) -> None:
    rnd = random.Random(seed)

    base_freq = rnd.choice([28, 32, 36, 40, 45])
    harm = base_freq * rnd.choice([2, 3, 4])

    drone = Sine(base_freq).to_audio_segment(duration=int(duration_s * 1000)).apply_gain(-10)
    drone2 = Sine(harm).to_audio_segment(duration=int(duration_s * 1000)).apply_gain(-18)

    noise = WhiteNoise().to_audio_segment(duration=int(duration_s * 1000)).apply_gain(-35)

    # slow “pumping” via chunk gain variation + mild clipping
    combined = drone.overlay(drone2).overlay(noise)
    chunks = []
    step_ms = 250
    for i in range(0, len(combined), step_ms):
        c = combined[i : i + step_ms]
        g = rnd.uniform(-2.5, 2.5) + (math_sin(i / 1000.0 * rnd.uniform(0.2, 0.55)) * 3.0)
        chunks.append(c.apply_gain(g))
    combined = sum(chunks)

    # distortion: overdrive then limit
    combined = combined.apply_gain(rnd.uniform(6, 10))
    combined = combined.compress_dynamic_range(threshold=-18.0, ratio=6.0, attack=5, release=80)
    combined = combined.apply_gain(-6)

    combined.export(out_wav, format="wav")


def math_sin(x: float) -> float:
    import math

    return math.sin(x)


def generate_tts_line(text: str, out_mp3: Path) -> bool:
    """
    gTTS needs internet access; in GitHub Actions it usually works.
    If it fails, we just skip narration.
    """
    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang="en", slow=False)
        tts.save(str(out_mp3))
        return out_mp3.exists() and out_mp3.stat().st_size > 1000
    except Exception:
        return False


def build_audio(duration_s: float, seed: int, out_wav: Path, headline: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        drone_wav = td / "drone.wav"
        generate_drone_audio(duration_s, seed=seed, out_wav=drone_wav)

        narration_mp3 = td / "tts.mp3"
        narration_ok = generate_tts_line(
            text=f"{headline}. The weather is fine. Do not leave the room.",
            out_mp3=narration_mp3,
        )

        drone = AudioSegment.from_file(drone_wav)
        mix = drone

        if narration_ok:
            nar = AudioSegment.from_file(narration_mp3)
            # make narration creepy: reduce pitch slightly by changing frame rate
            nar = nar._spawn(nar.raw_data, overrides={"frame_rate": int(nar.frame_rate * 0.92)}).set_frame_rate(
                nar.frame_rate
            )
            nar = nar.apply_gain(-10)
            # place narration around 10s
            mix = mix.overlay(nar, position=10_000)

        # final safety trim
        mix = mix[: int(duration_s * 1000)]
        mix.export(out_wav, format="wav")


def choose_assets(keyword: str, tmp: Path, seed: int) -> Tuple[List[Path], Optional[Path], str]:
    """
    Downloads multiple images; also downloads one special entity image.
    Returns (image_paths, entity_path, headline).
    """
    rnd = random.Random(seed)

    commons = scraper.fetch_commons_images(keyword, max_images=6)
    archive = scraper.fetch_archive_images(keyword, max_images=4)
    headline = scraper.pick_rss_headline(seed=seed)

    # Resolve archive dir -> direct file link
    resolved_archive = []
    for a in archive:
        direct = scraper.resolve_archive_download_dir(a.url)
        if direct:
            resolved_archive.append(scraper.ImageAsset(url=direct, source=a.source, title=a.title, license_hint=a.license_hint))

    pool = commons + resolved_archive
    rnd.shuffle(pool)
    if not pool:
        raise RuntimeError("No images found from Commons/Archive.")

    ensure_dir(tmp)

    img_paths: List[Path] = []
    for i, asset in enumerate(pool[:8]):
        p = tmp / f"img_{i}.jpg"
        ok = download_file(asset.url, p)
        if ok:
            img_paths.append(p)

    # entity frame: pick a different keyword for a “wrong” image
    entity_kw = rnd.choice(["face in window", "empty classroom", "security camera", "abandoned hospital corridor"])
    entity_assets = scraper.fetch_commons_images(entity_kw, max_images=4)
    rnd.shuffle(entity_assets)
    entity_path = None
    for j, a in enumerate(entity_assets):
        p = tmp / f"entity_{j}.jpg"
        if download_file(a.url, p):
            entity_path = p
            break

    return img_paths, entity_path, headline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="output.mp4")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    seed = args.seed if args.seed is not None else random.randint(1, 2_000_000_000)
    rnd = random.Random(seed)

    W, H = args.width, args.height
    duration = float(args.duration)
    fps = int(args.fps)

    keyword = scraper.pick_keyword(seed=seed)
    print(f"[seed={seed}] keyword='{keyword}'")

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        img_paths, entity_path, headline = choose_assets(keyword, tmp=td / "assets", seed=seed)

        if len(img_paths) < 2:
            raise RuntimeError("Not enough downloaded images to build montage.")

        # build montage segments summing to duration
        seg_count = min(8, max(4, len(img_paths)))
        base_seg = duration / seg_count
        durations = [base_seg for _ in range(seg_count)]
        # add tiny jitter
        durations = [max(4.0, d + rnd.uniform(-1.2, 1.2)) for d in durations]
        # normalize to exact duration
        s = sum(durations)
        durations = [d * (duration / s) for d in durations]

        clips: List[VideoClip] = []
        for i in range(seg_count):
            p = img_paths[i % len(img_paths)]
            img = load_image_as_rgb(p, (W, H))
            kb = make_kenburns_clip(img, dur=durations[i], fps=fps, seed=seed + i * 999)
            clips.append(kb)

        base = concatenate_videoclips(clips, method="compose").set_duration(duration).set_fps(fps)

        # VHS pass
        vhs = base.fl(lambda gf, t: effects.apply_vhs(gf(t), t=t, seed=seed))

        # overlays: lore + weird headline line
        overlays = make_lore_overlays(W, H, duration, headline=headline, seed=seed)

        # entity frame: 0.1s at random moment, low opacity
        if entity_path:
            entity_rgb = load_image_as_rgb(entity_path, (W, H))
            entity_t = rnd.uniform(8.0, max(9.0, duration - 6.0))
            entity_dur = 0.10

            def entity_frame(get_frame, t):
                fr = get_frame(t)
                if entity_t <= t <= (entity_t + entity_dur):
                    return effects.overlay_entity(fr, entity_rgb, opacity=0.12)
                return fr

            vhs = vhs.fl(lambda gf, t: entity_frame(gf, t))

        # Audio build
        audio_wav = td / "audio.wav"
        build_audio(duration, seed=seed, out_wav=audio_wav, headline=headline)

        audio_clip = AudioFileClip(str(audio_wav)).set_duration(duration)

        final = CompositeVideoClip([vhs, *overlays], size=(W, H)).set_duration(duration).set_fps(fps)
        final = final.set_audio(audio_clip)

        out_path = Path(args.out).resolve()
        print(f"Rendering to: {out_path}")
        final.write_videofile(
            str(out_path),
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            bitrate="2500k",
            preset="medium",
            threads=2,
            ffmpeg_params=[
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )

        print("Done.")


if __name__ == "__main__":
    main()
