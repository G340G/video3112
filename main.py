import os
import time
import random
import base64
import requests
import numpy as np
from io import BytesIO
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont

# MoviePy imports
from moviepy.editor import (ImageClip, concatenate_videoclips, AudioFileClip, 
                            CompositeAudioClip)
from moviepy.audio.AudioClip import AudioArrayClip

# Local imports
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
            return ImageFont.truetype(config.FONT_PATH, 60)
        except:
            return ImageFont.load_default()

    def scrape_uncanny_text(self):
        """
        Simulates scraping external data. 
        In a real scenario, you might hit a Wikipedia API or a news feed.
        Here we fetch a random line from a tech/philosophical source to be vague.
        """
        try:
            # Fetching random tech jargon can be eerie out of context
            response = requests.get("https://baconipsum.com/api/?type=meat-and-filler&sentences=1")
            if response.status_code == 200:
                return response.json()[0]
        except:
            pass
        return random.choice(config.CREEPY_PHRASES)

    def scrape_image(self):
        """Fetches a random abstract image to distort."""
        url = "https://picsum.photos/1280/720?grayscale&blur=2"
        response = requests.get(url)
        return Image.open(BytesIO(response.content))

    def generate_voice(self, text, filename, speed="slow"):
        """Generates the TTS audio."""
        tts = gTTS(text=text, lang='en', slow=(speed == "slow"))
        tts.save(filename)
        return AudioFileClip(filename)

    def generate_drone_sound(self, duration=5):
        """Generates a low-frequency horror drone sound (50Hz - 100Hz)."""
        sample_rate = 44100
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Create a low sine wave (60Hz hum)
        audio = 0.5 * np.sin(2 * np.pi * 60 * t)
        # Add random noise
        noise = np.random.normal(0, 0.1, audio.shape)
        final_audio = audio + noise
        
        # MoviePy expects stereo (2 channels)
        stereo_audio = np.vstack((final_audio, final_audio)).T
        return AudioArrayClip(stereo_audio, fps=sample_rate)

    def create_video(self):
        print(f"--- INITIATING BROADCAST: {self.entity} ---")

        # --- PART 1: THE WEATHER (NORMAL) ---
        img_normal = Image.new('RGB', config.VIDEO_RES, color=(10, 20, 100))
        d = ImageDraw.Draw(img_normal)
        d.text((100, 100), f"CHANNEL 5 WEATHER", fill=(255, 255, 255), font=self.font)
        d.text((100, 250), f"LOC: {self.location}", fill=(200, 200, 255), font=self.font)
        d.text((100, 350), f"TEMP: {self.temp} F", fill=(200, 200, 255), font=self.font)
        d.text((100, 550), "COMING UP: 60 MINUTE NEWS", fill=(255, 255, 0), font=self.font)
        
        clip_normal = ImageClip(np.array(img_normal)).set_duration(6)
        audio_normal = self.generate_voice(
            f"Good evening. It is a beautiful night in {self.location}. Temperatures are stable.", 
            "voice_normal.mp3", speed="normal"
        )
        clip_normal = clip_normal.set_audio(audio_normal)

        # --- PART 2: THE GLITCH ---
        clip_static = glitch_ops.create_static(duration=0.8)

        # --- PART 3: THE HORROR (INTERRUPTION) ---
        # Scrape and Distort
        raw_img = self.scrape_image()
        distorted_img = glitch_ops.vhs_distort(raw_img)
        
        # Overlay Scraped Text
        scraped_text = self.scrape_uncanny_text()
        d_horror = ImageDraw.Draw(distorted_img)
        d_horror.text((50, 600), scraped_text[:40], fill=(255, 0, 0), font=self.font)
        d_horror.text((50, 300), f"BEWARE: {self.entity}", fill=(255, 0, 0), font=self.font)
        
        clip_horror = ImageClip(np.array(distorted_img)).set_duration(5)
        
        # Audio: Drone + Creepy Voice
        voice_horror = self.generate_voice(
            f"Alert. {self.entity} has been sighted. {scraped_text}", 
            "voice_horror.mp3", speed="slow"
        )
        drone_sound = self.generate_drone_sound(duration=5)
        # Composite audio (Voice + Drone)
        comp_audio = CompositeAudioClip([voice_horror, drone_sound.set_volume(0.3)])
        clip_horror = clip_horror.set_audio(comp_audio)

        # --- PART 4: ARG ENCRYPTION ---
        # Base64 Encode the truth
        encoded_truth = base64.b64encode(config.HIDDEN_TRUTH.encode()).decode()
        
        img_code = Image.new('RGB', config.VIDEO_RES, color=(0, 0, 0))
        d_code = ImageDraw.Draw(img_code)
        d_code.text((150, 360), f"HEX: {encoded_truth}", fill=(0, 255, 0), font=self.font)
        
        clip_code = ImageClip(np.array(img_code)).set_duration(3)

        # --- RENDER ---
        final_video = concatenate_videoclips([
            clip_normal, 
            clip_static, 
            clip_horror, 
            clip_static.set_duration(0.2), # Jumpscare flash
            clip_code
        ])
        
        output_file = f"broadcast_{int(time.time())}.mp4"
        final_video.write_videofile(output_file, fps=config.FPS)
        
        # Cleanup temp audio
        if os.path.exists("voice_normal.mp3"): os.remove("voice_normal.mp3")
        if os.path.exists("voice_horror.mp3"): os.remove("voice_horror.mp3")

if __name__ == "__main__":
    bot = BroadcastGenerator()
    bot.create_video()
