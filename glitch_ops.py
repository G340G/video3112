import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from moviepy.editor import ImageClip
import random

def create_static(duration=1, size=(1280, 720)):
    """Generates TV static noise."""
    def make_frame(t):
        return np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    return ImageClip(make_frame, duration=duration)

def vhs_distort(pil_image):
    """
    Applies a 'VHS' style degradation:
    1. Chromatic Aberration (RGB Split)
    2. Blur
    3. Noise
    """
    # 1. Chromatic Aberration
    r, g, b = pil_image.split()
    # Shift channels slightly to create the "3D glasses" look
    width, height = pil_image.size
    
    # Crop and resize to shift
    r = r.transform((width, height), Image.AFFINE, (1, 0, 3, 0, 1, 0))
    b = b.transform((width, height), Image.AFFINE, (1, 0, -3, 0, 1, 0))
    
    # Merge back
    img = Image.merge("RGB", (r, g, b))
    
    # 2. Gaussian Blur (The "Old Tape" look)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    
    # 3. Add Noise
    img_arr = np.array(img)
    noise = np.random.normal(0, 15, img_arr.shape).astype(np.uint8)
    noise_img = img_arr + noise
    
    # Clip values to ensure valid image data
    final_img = Image.fromarray(np.clip(noise_img, 0, 255).astype(np.uint8))
    
    # Darken slightly for horror vibe
    enhancer = ImageEnhance.Brightness(final_img)
    final_img = enhancer.enhance(0.7)
    
    return final_img
