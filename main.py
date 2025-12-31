import os
import time
import random
import base64
import subprocess
from io import BytesIO

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip, CompositeAudioClip
from moviepy.audio.AudioClip import AudioArrayClip
import moviepy.audio.fx.all as afx

import config
import glitch_ops


class BroadcastGenerator:
    def __init__(self):
        self.entity = random.choice(config.ENTITIES)
        self.location = random.choice(config.LOCATIONS)
        self.weather = random.choice(config.WEATHER_CONDITIONS)
        self.temp = random.randint(30, 99)
        self.font = self._load_font()

    def _load_font(self):
        """Loads a VCR font if available, otherwise default."""
        try:
            if os.path.exists(config.FONT_PATH):
                return ImageFont.truetype(config.FONT_PATH, 42)
        except Exception:
            pass
        return ImageFont.load_default()

    def scrape_uncanny_text(self):
        """Fetches an unsettling phrase. Falls back to local phrases."""
        try:
            r = requests.get("https://www.quotable.io/random?tags=famous-quotes", timeout=8)
            if r.ok:
                data = r.json()
                txt = (data.get("content") or "").strip()
                if txt:
                    return txt.upper()
        except Exception:
            pass
        return random.choice(config.CREEPY_PHRASES)

    def scrape_image(self):
        """Fetches a random abstract image to distort. Falls back to generated noise."""
        url = "https://picsum.photos/1280/720?grayscale&blur=2"
        try:
            r = requests.get(url, timeout=12)
            r.raise_for_status()
            return Image.open(BytesIO(r.content)).convert("RGB")
        except Exception:
            w, h = config.VIDEO_RES
            arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            return Image.fromarray(arr, "RGB")

    def _voice_espeak(self, text: str, out_path: str, speed: str = "normal") -> bool:
        """Offline TTS via espeak-ng/espeak. Returns True on success."""
        wpm = "165" if speed == "normal" else "125"
        voice = "en-us"
        for exe in ("espeak-ng", "espeak"):
            try:
                subprocess.run(
                    [exe, "-v", voice, "-s", wpm, "-w", out_path, text],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return os.path.exists(out_path) and os.path.getsize(out_path) > 1000
            except Exception:
                continue
        return False

    def fit_audio(self, audio, target_duration):
        """Ensures an AudioClip is at least target_duration long (loops if needed)."""
        if target_duration is None:
            return audio
        try:
            dur = getattr(audio, "duration", None)
            if dur is None:
                return audio.set_duration(target_duration)
            if dur + 0.02 < target_duration:
                return afx.audio_loop(audio, duration=target_duration)
            return audio.set_duration(target_duration)
        except Exception:
            try:
                return audio.set_duration(target_duration)
            except Exception:
                return audio

    def generate_radio_tone(self, duration=6):
        """A synthetic 'broadcast tone' (no network, no external binaries)."""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        carrier = 0.35 * np.sin(2 * np.pi * 220 * t)
        trem = (0.5 + 0.5 * np.sin(2 * np.pi * 4.2 * t))
        hiss = np.random.normal(0, 0.12, carrier.shape)
        audio = (carrier * trem) + hiss

        stereo = np.vstack((audio, audio)).T.astype(np.float32)
        return AudioArrayClip(stereo, fps=sample_rate)

    def generate_voice(self, text, filename, speed="slow", target_duration=None):
        """Generates TTS audio with robust fallbacks.

        1) Prefer offline espeak (reliable on GitHub Actions).
        2) If espeak is unavailable, attempt gTTS (network required).
        3) Final fallback: synthetic radio tone/noise.
        """
        base, _ = os.path.splitext(filename)
        wav_path = base + ".wav"

        # 1) Offline TTS
        if self._voice_espeak(text, wav_path, speed=speed):
            audio = AudioFileClip(wav_path)
            return self.fit_audio(audio, target_duration)

        # 2) gTTS (optional)
        try:
            from gtts import gTTS  # optional dependency
            tts = gTTS(text=text, lang="en", slow=(speed == "slow"))
            tts.save(filename)
            audio = AudioFileClip(filename)
            return self.fit_audio(audio, target_duration)
        except Exception:
            pass

        # 3) Hard fallback
        duration = max(3, min(10, int(len(text) / 12) + 3))
        return self.fit_audio(self.generate_radio_tone(duration=duration), target_duration)

    def generate_drone_sound(self, duration=5):
        """Generates a low-frequency horror drone sound (50Hz - 100Hz)."""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

        audio = 0.5 * np.sin(2 * np.pi * 60 * t)
        noise = np.random.normal(0, 0.1, audio.shape)
        final_audio = audio + noise

        stereo_audio = np.vstack((final_audio, final_audio)).T.astype(np.float32)
        return AudioArrayClip(stereo_audio, fps=sample_rate)

    def _cleanup_temp_audio(self):
        for p in ("voice_normal.mp3", "voice_horror.mp3", "voice_normal.wav", "voice_horror.wav"):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

    def create_video(self):
        print(f"--- INITIATING BROADCAST: {self.entity} ---")

        # --- PART 1: THE NORMAL ---
        img_normal = Image.new("RGB", config.VIDEO_RES, color=(10, 10, 20))
        d = ImageDraw.Draw(img_normal)
        d.text((50, 50), "LOCAL NEWS NETWORK", fill=(0, 255, 0), font=self.font)
        d.text((50, 120), f"LOCATION: {self.location}", fill=(0, 255, 0), font=self.font)
        d.text((50, 190), f"WEATHER: {self.weather}", fill=(0, 255, 0), font=self.font)
        d.text((50, 260), f"TEMP: {self.temp}F", fill=(0, 255, 0), font=self.font)
        d.text((50, 360), "DO NOT ADJUST YOUR SET", fill=(0, 255, 0), font=self.font)

        clip_normal = ImageClip(np.array(img_normal)).set_duration(5)
        audio_normal = self.generate_voice(
            f"This is an emergency broadcast for {self.location}. Weather advisory: {self.weather}.",
            "voice_normal.mp3",
            speed="normal",
            target_duration=clip_normal.duration,
        )
        clip_normal = clip_normal.set_audio(audio_normal)

        # --- PART 2: STATIC ---
        clip_static = glitch_ops.create_static(duration=0.8, size=config.VIDEO_RES)

        # --- PART 3: THE INTERRUPTION ---
        raw_img = self.scrape_image()
        distorted_img = glitch_ops.vhs_distort(raw_img)

        scraped_text = self.scrape_uncanny_text()
        d_horror = ImageDraw.Draw(distorted_img)
        d_horror.text((50, 600), scraped_text[:44], fill=(255, 0, 0), font=self.font)

        clip_horror = ImageClip(np.array(distorted_img)).set_duration(6)
        audio_drone = self.fit_audio(self.generate_drone_sound(duration=6), clip_horror.duration)
        audio_horror_voice = self.generate_voice(
            f"{random.choice(config.CREEPY_PHRASES)}. {random.choice(config.CREEPY_PHRASES)}.",
            "voice_horror.mp3",
            speed="slow",
            target_duration=clip_horror.duration,
        )
        clip_horror = clip_horror.set_audio(
            CompositeAudioClip([audio_drone, audio_horror_voice]).set_duration(clip_horror.duration)
        )

        # --- PART 4: CODED MESSAGE ---
        img_code = Image.new("RGB", config.VIDEO_RES, color=(0, 0, 0))
        d_code = ImageDraw.Draw(img_code)

        truth = config.HIDDEN_TRUTH.encode("utf-8")
        encoded_truth = base64.b64encode(truth).decode("utf-8")

        d_code.text((80, 150), "== SIGNAL FRAGMENT ==", fill=(0, 255, 0), font=self.font)
        d_code.text((80, 230), f"ENTITY: {self.entity}", fill=(0, 255, 0), font=self.font)
        d_code.text((80, 300), f"HEX: {encoded_truth}", fill=(0, 255, 0), font=self.font)
        d_code.text((80, 380), "DECODE. REPEAT. OBEY.", fill=(0, 255, 0), font=self.font)

        clip_code = ImageClip(np.array(img_code)).set_duration(3)

        final_video = concatenate_videoclips(
            [clip_normal, clip_static, clip_horror, clip_static.set_duration(0.2), clip_code],
            method="compose",
        )

        output_file = f"broadcast_{int(time.time())}.mp4"
        final_video.write_videofile(
            output_file,
            fps=config.FPS,
            codec="libx264",
            audio_codec="aac",
            threads=4,
            preset="medium",
            verbose=False,
            logger=None,
        )

        self._cleanup_temp_audio()
        print(f"--- BROADCAST COMPLETE: {output_file} ---")


if __name__ == "__main__":
    BroadcastGenerator().create_video()
