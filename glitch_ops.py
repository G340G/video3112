import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from moviepy.editor import ImageClip, VideoClip # Added VideoClip
import random

def create_static(duration=1, size=(1280, 720)):
    """Generates TV static noise compatible with MoviePy 1.0.3."""
    # Create a single frame of noise
    noise_array = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    
    # Return as an ImageClip set to the specific duration
    return ImageClip(noise_array).set_duration(duration)

def vhs_distort(pil_image):
    """Applies a 'VHS' style degradation."""
    # 1. Chromatic Aberration
    r, g, b = pil_image.split()
    width, height = pil_image.size
    
    # Shift channels
    r = r.transform((width, height), Image.AFFINE, (1, 0, 3, 0, 1, 0))
    b = b.transform((width, height), Image.AFFINE, (1, 0, -3, 0, 1, 0))
    
    img = Image.merge("RGB", (r, g, b))
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # 3. Add Noise
    img_arr = np.array(img)
    noise = np.random.normal(0, 15, img_arr.shape).astype(np.uint8)
    final_img = Image.fromarray(np.clip(img_arr + noise, 0, 255).astype(np.uint8))
    
    enhancer = ImageEnhance.Brightness(final_img)
    return enhancer.enhance(0.7)
