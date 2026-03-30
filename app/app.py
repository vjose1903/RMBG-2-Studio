# Standard library imports
import os
import re
import sys
import glob
import warnings
import subprocess
from datetime import datetime
from pathlib import Path
from io import BytesIO
from threading import Lock

# Third-party imports
import cv2
import torch
import gradio as gr
import numpy as np
import requests
import colorsys
from tqdm import tqdm
import devicetorch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
from gradio_imageslider import ImageSlider
from loadimg import load_img  # Image loading utility

# change httx/httpcore default from INFO to stop its startup log interferring with URL capture
import logging
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)  

# ML/AI framework imports
from transformers import AutoModelForImageSegmentation  # Hugging Face model for background removal

# Configure warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='timm')

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output_images"
MODEL_ID = os.environ.get("RMBG_MODEL_ID", "cocktailpeanut/rm")
HOST = os.environ.get("RMBG_HOST", "127.0.0.1")
PORT = int(os.environ.get("RMBG_PORT", "7860"))
DEVICE_PREFERENCE = os.environ.get("RMBG_DEVICE", "auto").lower()
torch.set_float32_matmul_precision(["high", "highest"][0])

birefnet = None
model_lock = Lock()


def pick_device():
    if DEVICE_PREFERENCE == "cpu":
        return "cpu"

    if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
        return "cuda"

    if DEVICE_PREFERENCE == "mps" and torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


device = pick_device()
MODEL_INPUT_SIZE = int(os.environ.get("RMBG_INPUT_SIZE", "768" if device == "mps" else "1024"))

transform_image = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


MAX_GALLERY_IMAGES = 1000
output_folder = str(OUTPUT_DIR)



OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def is_output_image(path_like):
    try:
        return Path(path_like).resolve().is_relative_to(OUTPUT_DIR.resolve())
    except Exception:
        return False


def get_model():
    global birefnet

    if birefnet is not None:
        return birefnet

    with model_lock:
        if birefnet is None:
            print(f"Loading model: {MODEL_ID} on {device} (input {MODEL_INPUT_SIZE}px)", flush=True)
            birefnet = AutoModelForImageSegmentation.from_pretrained(
                MODEL_ID, trust_remote_code=True
            )
            birefnet = birefnet.to(device)

    return birefnet


def empty_device_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    devicetorch.empty_cache(torch)


def move_model_to(target_device):
    global birefnet, device

    device = target_device
    if birefnet is not None:
        birefnet = birefnet.to(target_device)


def is_mps_oom_error(error):
    message = str(error).lower()
    return "mps backend out of memory" in message or ("out of memory" in message and "mps" in message)

def generate_filename(prefix="no_bg"):
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S_%f")
    return f"{prefix}_{timestamp}.png"

def open_output_folder():
    folder_path = os.path.abspath(output_folder)
    try:
        if os.name == 'nt':  # Windows
            subprocess.run(['explorer', folder_path])
        elif sys.platform == 'darwin':  # macOS
            subprocess.run(['open', folder_path])
        else:  # Linux
            subprocess.run(['xdg-open', folder_path])
        return "✅ Opened outputs folder"
    except Exception as e:
        return f"❌ Error opening folder: {str(e)}"


def is_valid_image_url(url):
    """Validate if the URL points to an image file."""
    try:
        # Check if URL pattern is valid
        if not re.match(r'https?://.+', url):
            return False
        
        # Some CDNs reject HEAD, so use a streamed GET and read headers only.
        response = requests.get(url, timeout=10, stream=True)
        content_type = response.headers.get('content-type', '').lower()
        response.close()
        return (response.status_code == 200 and 
                any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/gif', 'image/webp']))
    except requests.ConnectionError:
        raise ConnectionError("Unable to connect. Please check your internet connection")
    except requests.Timeout:
        raise TimeoutError("Request timed out. The server took too long to respond")
    except:
        raise ValueError("Failed to validate URL")

def download_image_from_url(url):
    """Download image from URL and return as PIL Image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.ConnectionError:
        raise ConnectionError("Unable to connect. Please check your internet connection")
    except requests.Timeout:
        raise TimeoutError("Request timed out. The server took too long to respond")
    except requests.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError("Image not found (404 error)")
        elif e.response.status_code == 403:
            raise ValueError("Access to image denied (403 error)")
        else:
            raise ValueError(f"HTTP error occurred (Status code: {e.response.status_code})")
    except Exception as e:
        raise ValueError(f"Failed to download image: {str(e)}")
        
def process_input(input_data):
    """Process either uploaded image or URL input."""
    try:
        if isinstance(input_data, str) and input_data.strip():
            # Handle URL input
            url = input_data.strip()
            try:
                if not is_valid_image_url(url):
                    return None, "❌ Invalid image URL. Please ensure the URL directly links to an image (jpg, png, gif, or webp)"
                image = download_image_from_url(url)
                return image, "✅ Successfully downloaded and processed image from URL"
            except ConnectionError:
                return None, "❌ No internet connection. Please check your network and try again"
            except TimeoutError:
                return None, "❌ Connection timed out. The server took too long to respond"
            except ValueError as e:
                return None, f"❌ {str(e)}"
        else:
            # Handle direct image upload
            image = load_img(input_data, output_type="pil")
            return image, None  # None means don't update status for regular uploads
    except Exception as e:
        return None, f"❌ Error: {str(e)}"
        
        
def batch_process_images(files, progress=gr.Progress()):
    """Process multiple images with enhanced error handling and validation"""
    if not files:
        return "⚠️ No files selected. Please upload some images to process.", None
    
    results = {
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'processed_files': [],
        'error_files': []
    }
    
    # Valid image extensions (case-insensitive)
    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
    
    try:
        total_files = len(files)
        for i, file in enumerate(files):
            try:
                # Update progress bar
                progress(i/total_files, f"Processing {i+1}/{total_files}")
                
                # Check file extension
                file_ext = os.path.splitext(file.name)[1].lower()
                if file_ext not in valid_extensions:
                    results['skipped'] += 1
                    results['error_files'].append(f"{os.path.basename(file.name)} (Unsupported format)")
                    continue
                
                # Load and process image
                img = load_img(file.name, output_type="pil")
                img = img.convert("RGB")
                processed = process(img)
                
                # Save with original filename plus suffix
                original_name = Path(file.name).stem
                new_filename = f"{original_name}_nobg.png"
                output_path = os.path.join(output_folder, new_filename)
                processed.save(output_path)
                
                results['successful'] += 1
                results['processed_files'].append(new_filename)
                
            except Exception as e:
                results['failed'] += 1
                results['error_files'].append(f"{os.path.basename(file.name)} ({str(e)})")
                
        # Prepare detailed status message
        status_parts = [
            "✅ Processing complete!",
            f"Successfully processed: {results['successful']} images",
        ]
        
        if results['skipped'] > 0:
            status_parts.append(f"Skipped: {results['skipped']} files (unsupported format)")
        
        if results['failed'] > 0:
            status_parts.append(f"Failed: {results['failed']} images")
            
        if results['error_files']:
            status_parts.append("\nDetails of skipped/failed files:")
            status_parts.extend(f"- {err}" for err in results['error_files'])
            
        status_parts.append(f"\nOutput saved to: {output_folder}")
        
        return "\n".join(status_parts), update_gallery()
                
    except Exception as e:
        return f"❌ Unexpected error during batch processing: {str(e)}", update_gallery()
        
        
def fn(image_input):
    if image_input is None:
        return None, update_gallery(), "⚠️ No image provided"
    
    image, status_msg = process_input(image_input)
    if image is None:
        return None, update_gallery(), status_msg
    
    origin = image.copy()
    try:
        processed_image = process(image)
    except Exception as e:
        return None, update_gallery(), f"❌ Processing failed: {str(e)}"
    unique_filename = generate_filename()
    image_path = os.path.join(output_folder, unique_filename)
    processed_image.save(image_path)
    gallery_paths = update_gallery()
    
    # Return status message only for URL processing
    return (processed_image, origin), gallery_paths, status_msg
    
    
def process(image):
    model = get_model()
    rgb_image = image.convert("RGB")
    image_size = rgb_image.size
    input_images = transform_image(rgb_image).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
    except RuntimeError as e:
        if device == "mps" and is_mps_oom_error(e):
            print("MPS ran out of memory. Retrying on CPU.", flush=True)
            empty_device_cache()
            move_model_to("cpu")
            model = get_model()
            input_images = transform_image(rgb_image).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(input_images)[-1].sigmoid().cpu()
        else:
            raise

    pred = preds[0].squeeze()
    pred_pil = transforms.ToPILImage()(pred)
    mask = pred_pil.resize(image_size)
    result = rgb_image.copy()
    result.putalpha(mask)
    empty_device_cache()
    return result


# Gallery management
gallery_paths = []

def update_gallery():
    """Update gallery with most recent images, limited to prevent UI overload"""
    global gallery_paths
    all_images = [
        os.path.join(output_folder, f) 
        for f in os.listdir(output_folder) 
        if f.endswith(".png")
    ]
    # Sort by file modification time, newest first
    all_images.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    # Limit to most recent images
    gallery_paths = all_images[:MAX_GALLERY_IMAGES]
    return gallery_paths


def combine_images(fg_path, bg_path, scale, x_offset=0, y_offset=0, flip_h=False, flip_v=False, 
                  rotation=0, brightness=1.0, contrast=1.0, saturation=1.0, 
                  temperature=0, tint_color=None, tint_strength=0):
    if not (fg_path and bg_path):
        return None

    # Process foreground image
    if isinstance(fg_path, str) and is_output_image(fg_path):
        fg = Image.open(fg_path)
    elif isinstance(fg_path, Image.Image):
        fg = fg_path.copy()
        if fg.mode != 'RGBA':
            fg = process(fg)
    else:
        fg = load_img(fg_path, output_type="pil")
        fg = process(fg)
    
    # Apply color adjustments to foreground
    fg = apply_color_adjustments(
        fg, brightness, contrast, saturation,
        temperature, tint_color, tint_strength
    )
    
    bg = Image.open(bg_path) if isinstance(bg_path, str) else bg_path
    
    if fg.mode != 'RGBA':
        fg = fg.convert('RGBA')
    
    bg = bg.convert('RGBA')
    
    # Apply transformations
    if flip_h:
        fg = fg.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_v:
        fg = fg.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if rotation:
        fg = fg.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)
    
    new_width = int(fg.size[0] * (scale / 100))
    new_height = int(fg.size[1] * (scale / 100))
    fg = fg.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    center_x = (bg.size[0] - new_width) // 2
    center_y = (bg.size[1] - new_height) // 2
    
    x_pos = center_x + x_offset
    y_pos = center_y - y_offset
    
    result = bg.copy()
    result.paste(fg, (x_pos, y_pos), fg)
    
    return result


def calculate_fit_scale(fg_image, bg_image):
    """Calculate scale percentage to fit foreground within background"""
    if not (fg_image and bg_image):
        return 100
        
    # Get image sizes
    if isinstance(fg_image, str):
        fg_image = Image.open(fg_image)
    if isinstance(bg_image, str):
        bg_image = Image.open(bg_image)
    
    # Calculate ratios
    width_ratio = bg_image.width / fg_image.width
    height_ratio = bg_image.height / fg_image.height
    
    # Use the smaller ratio to ensure fit
    fit_ratio = min(width_ratio, height_ratio)
    
    # Convert to percentage, with a small margin
    return int(fit_ratio * 95)  # 95% of perfect fit to leave a margin


def adjust_color_temperature(image, temperature):
    """Adjust color temperature of an image (negative=cool, positive=warm)"""
    # Convert to numpy array for processing
    img_array = np.array(image)
    
    # Separate the alpha channel if it exists
    has_alpha = img_array.shape[-1] == 4
    if has_alpha:
        img_rgb = img_array[..., :3]
        alpha = img_array[..., 3]
    else:
        img_rgb = img_array
    
    # Adjust temperature by modifying RGB channels
    if temperature > 0:  # Warmer
        img_rgb[..., 0] = np.clip(img_rgb[..., 0] + temperature, 0, 255)  # More red
        img_rgb[..., 2] = np.clip(img_rgb[..., 2] - temperature/2, 0, 255)  # Less blue
    else:  # Cooler
        img_rgb[..., 2] = np.clip(img_rgb[..., 2] - temperature, 0, 255)  # More blue
        img_rgb[..., 0] = np.clip(img_rgb[..., 0] + temperature/2, 0, 255)  # Less red
    
    # Recombine with alpha if necessary
    if has_alpha:
        img_array = np.dstack((img_rgb, alpha))
    else:
        img_array = img_rgb
    
    return Image.fromarray(img_array.astype('uint8'))


def apply_color_adjustments(image, brightness=1.0, contrast=1.0, saturation=1.0, 
                          temperature=0, tint_color=None, tint_strength=0):
    """Apply color adjustments to an image while preserving transparency"""
    if image is None:
        return None
        
    # Store original alpha channel
    alpha = None
    if image.mode == 'RGBA':
        alpha = image.split()[3]
    
    # Convert to RGB for adjustments
    img = image.convert('RGB')
    
    # Apply basic adjustments
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if temperature != 0:
        img = adjust_color_temperature(img, temperature)
    
    # Apply tint if specified
    if tint_color and tint_strength > 0:
        tint_layer = Image.new('RGB', img.size, tint_color)
        img = Image.blend(img, tint_layer, tint_strength)
    
    # Restore alpha channel if it existed
    if alpha:
        img.putalpha(alpha)
    
    return img


def update_preview(fg, bg, scale, x, y, rotation, flip_h, flip_v, 
                  brightness, contrast, saturation, temperature, 
                  tint_color, tint_strength):
    if not fg or not bg:
        return None
    return combine_images(
        fg, bg, scale, x, y, flip_h, flip_v, rotation,
        brightness, contrast, saturation, temperature, 
        tint_color, tint_strength
    )
                
def reset_controls():
    return 100, 0, 0, 0, False, False

def reset_color_controls():
    """Reset all color grading controls to default values"""
    return 1.0, 1.0, 1.0, 0, "#000000", 0

def handle_fg_change(fg, bg, *current_values):
    """
    Wrapper function to handle foreground image changes with control resets
    Returns the new image with default control values
    """
    # Get default values
    default_placement = reset_controls()
    default_colors = reset_color_controls()
    
    # If we have a foreground image, create preview with default values
    if fg is not None:
        preview = combine_images(
            fg, bg,
            scale=default_placement[0],          # 100
            x_offset=default_placement[1],       # 0
            y_offset=default_placement[2],       # 0
            rotation=default_placement[3],       # 0
            flip_h=default_placement[4],         # False
            flip_v=default_placement[5],         # False
            brightness=default_colors[0],        # 1.0
            contrast=default_colors[1],          # 1.0
            saturation=default_colors[2],        # 1.0
            temperature=default_colors[3],       # 0
            tint_color=default_colors[4],        # "#000000"
            tint_strength=default_colors[5]      # 0
        )
    else:
        preview = None
        
    # Return all values: preview image, placement controls, color controls
    return (
        preview,
        *default_placement,
        *default_colors
    )

    
def save_combined(image):
    if image is None:
        return update_gallery(), "⚠️ No image to save"
        
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
        
    output_path = os.path.join(output_folder, generate_filename("combined"))
    image.save(output_path)
    return update_gallery(), f"✅ Saved image: {os.path.basename(output_path)}"


css = """
:root {
    --ink: #f5f7fb;
    --muted: #9ca8bc;
    --paper: rgba(18, 27, 42, 0.92);
    --paper-strong: rgba(24, 35, 54, 0.96);
    --accent: #ee7b52;
    --accent-deep: #c65b36;
    --accent-soft: rgba(238, 123, 82, 0.16);
    --forest: #8fd3be;
    --line: rgba(158, 177, 210, 0.12);
    --shadow: 0 26px 60px rgba(3, 8, 18, 0.42);
    --radius-xl: 28px;
    --radius-lg: 22px;
    --radius-md: 16px;
    --heading-font: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
    --ui-font: "Avenir Next", "Segoe UI", "Trebuchet MS", sans-serif;
}

body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(238, 123, 82, 0.18), transparent 22%),
        radial-gradient(circle at top right, rgba(83, 118, 173, 0.18), transparent 28%),
        linear-gradient(180deg, #070b13 0%, #0d1421 46%, #111a29 100%) !important;
    color: var(--ink) !important;
    font-family: var(--ui-font) !important;
}

.gradio-container {
    max-width: 1500px !important;
    padding: 28px 20px 56px !important;
}

#app-shell {
    position: relative;
}

#app-shell::before,
#app-shell::after {
    content: "";
    position: absolute;
    z-index: 0;
    border-radius: 999px;
    filter: blur(10px);
    opacity: 0.65;
}

#app-shell::before {
    width: 240px;
    height: 240px;
    top: 18px;
    right: 18px;
    background: rgba(238, 123, 82, 0.10);
}

#app-shell::after {
    width: 320px;
    height: 320px;
    left: -90px;
    bottom: 80px;
    background: rgba(83, 118, 173, 0.12);
}

#app-shell > .gap {
    position: relative;
    z-index: 1;
}

.section-card,
.gallery-shell,
.info-card,
.control-card,
.status-card,
.action-card {
    background: var(--paper);
    border: 1px solid var(--line);
    box-shadow: var(--shadow);
    backdrop-filter: blur(16px);
}

.section-head {
    margin: 0 0 14px;
}

.section-head p {
    margin: 0 0 6px;
    font-size: 0.74rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--accent-deep);
}

.section-head h2,
.section-head h3 {
    margin: 0;
    font-family: var(--heading-font);
    font-size: 1.9rem;
    line-height: 1.04;
    letter-spacing: -0.03em;
    color: var(--ink);
}

.section-head .subcopy {
    margin-top: 8px;
    font-family: var(--ui-font);
    font-size: 0.98rem;
    line-height: 1.65;
    color: var(--muted);
}

.gallery-shell {
    border-radius: var(--radius-xl);
    padding: 22px;
}

.gallery-shell .wrap {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.gallery-shell .grid-wrap,
.gallery-shell .grid-container {
    border-radius: 20px !important;
}

.studio-tabs {
    background: transparent !important;
}

.studio-tabs > .tab-nav {
    gap: 10px !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
}

.studio-tabs button {
    border: 1px solid rgba(51, 41, 31, 0.10) !important;
    background: rgba(16, 24, 38, 0.9) !important;
    color: var(--muted) !important;
    border-radius: 999px !important;
    min-height: 48px !important;
    padding: 0 18px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

.studio-tabs button.selected {
    background: linear-gradient(135deg, var(--accent), var(--accent-deep)) !important;
    color: #fff8f2 !important;
    border-color: transparent !important;
}

.section-card,
.info-card,
.control-card,
.status-card,
.action-card {
    border-radius: var(--radius-xl);
}

.section-card {
    padding: 22px;
}

.info-card {
    padding: 18px 20px;
}

.info-card h3 {
    margin: 0 0 10px;
    font-family: var(--heading-font);
    font-size: 1.45rem;
    line-height: 1.1;
    color: var(--ink);
}

.info-card p,
.info-card li {
    color: var(--muted);
    line-height: 1.65;
}

.info-card ul {
    margin: 0;
    padding-left: 18px;
}

.media-stage {
    gap: 18px !important;
}

.media-panel,
.preview-panel {
    padding: 18px;
    border-radius: var(--radius-lg);
    background: linear-gradient(180deg, rgba(22, 31, 48, 0.96), rgba(17, 25, 40, 0.94));
    border: 1px solid rgba(158, 177, 210, 0.10);
}

.preview-panel {
    background:
        linear-gradient(180deg, rgba(19, 28, 43, 0.96), rgba(15, 23, 36, 0.96)),
        radial-gradient(circle at top right, rgba(238, 123, 82, 0.12), transparent 38%);
}

.image-container .image-custom,
.image-container .image-slider-custom {
    border-radius: 20px !important;
    overflow: hidden !important;
}

.image-container .image-custom {
    max-width: 100% !important;
    max-height: 78vh !important;
    width: auto !important;
    height: auto !important;
}

.image-container .image-slider-custom {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

.image-container .image-slider-custom > div {
    width: 100% !important;
    max-width: 100% !important;
    max-height: 78vh !important;
}

.image-container .image-slider-custom img {
    max-height: 78vh !important;
    width: 100% !important;
    height: auto !important;
    object-fit: contain !important;
}

.image-container .image-slider-custom .image-slider-handle {
    width: 3px !important;
    background: rgba(255, 249, 243, 0.95) !important;
    border: 2px solid rgba(48, 84, 75, 0.55) !important;
}

.preview-row {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
}

.control-card {
    padding: 8px 10px 14px;
}

.control-card .label-wrap > label,
.status-card .label-wrap > label,
.gallery-shell .label-wrap > label {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: var(--accent-deep) !important;
}

.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: var(--ink) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .scroll-hide {
    color: var(--ink) !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-input,
.gradio-container .gr-file,
.gradio-container .gr-form {
    background-color: rgba(10, 17, 28, 0.9) !important;
    border-color: rgba(158, 177, 210, 0.12) !important;
}

.gradio-container .gr-file,
.gradio-container .gr-box {
    color: var(--ink) !important;
}

.control-actions,
.action-row {
    gap: 12px !important;
}

.action-card {
    padding: 16px;
}

.status-card {
    padding: 14px 16px;
    background: linear-gradient(180deg, rgba(15, 24, 37, 0.95), rgba(18, 29, 45, 0.98));
}

.status-card textarea,
.status-card input {
    color: var(--forest) !important;
    font-weight: 500 !important;
}

button.primary,
.gr-button-primary {
    background: linear-gradient(135deg, var(--accent), var(--accent-deep)) !important;
    color: #fff8f2 !important;
    border: none !important;
    box-shadow: 0 14px 26px rgba(159, 67, 40, 0.24) !important;
}

button.secondary {
    background: rgba(17, 25, 39, 0.92) !important;
}

.tips-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
}

.tip-box {
    padding: 14px 16px;
    border-radius: 18px;
    background: rgba(11, 18, 29, 0.85);
    border: 1px solid rgba(158, 177, 210, 0.10);
}

.tip-box strong {
    display: block;
    margin-bottom: 6px;
    font-size: 0.84rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--accent-deep);
}

.tip-box span {
    color: var(--muted);
    line-height: 1.55;
    font-size: 0.95rem;
}
"""

    
# Create interface
with gr.Blocks(css=css, elem_id="app-shell") as demo:
    with gr.Tabs(selected="cutout", elem_classes=["studio-tabs"]) as tabs:
        with gr.Tab("History", id="history"):
            with gr.Column(elem_classes=["gallery-shell"]):
                gr.HTML(f"""
                <div class="section-head">
                  <p>Output Vault</p>
                  <h2>Recent Exports</h2>
                  <div class="subcopy">Your latest PNG renders live here. The gallery shows up to the most recent {MAX_GALLERY_IMAGES:,} images.</div>
                </div>
                """)
                shared_gallery = gr.Gallery(
                    label="Recent renders",
                    columns=5,
                    rows=3,
                    height="auto",
                    allow_preview=True,
                    preview=True,
                    object_fit="scale-down",
                    value=update_gallery()
                )

        with gr.Tab("Cutout Studio", id="cutout"):
            with gr.Column(elem_classes=["section-card"]):
                gr.HTML("""
                <div class="section-head">
                  <p>Quick Remove</p>
                  <h3>Upload an image and remove the background</h3>
                  <div class="subcopy">The first thing you see is the actual tool. Drop an image or paste a direct URL to generate a transparent cutout.</div>
                </div>
                """)
                with gr.Row(elem_classes=["media-stage"]):
                    with gr.Column(elem_classes=["media-panel", "image-container"]):
                        image = gr.Image(
                            type="pil",
                            label="Source Image",
                            elem_classes=["image-custom"]
                        )
                    with gr.Column(elem_classes=["preview-panel", "image-container"]):
                        slider1 = ImageSlider(
                            interactive=False,
                            label="Before / After",
                            elem_classes=["image-slider-custom"]
                        )
                with gr.Row(elem_classes=["action-row"]):
                    url_input = gr.Textbox(
                        label="Image URL",
                        placeholder="Paste a direct .jpg, .png, .gif, or .webp URL",
                        info="Use this if you want to load an image straight from the web."
                    )
                with gr.Row(elem_classes=["action-row"]):
                    with gr.Column(scale=7, elem_classes=["status-card"]):
                        status_text_1 = gr.Textbox(label="Session Status", interactive=False, lines=2)
                    with gr.Column(scale=2, elem_classes=["action-card"]):
                        open_folder_btn_1 = gr.Button("Open Output Folder", size="lg")

            open_folder_btn_1.click(open_output_folder, outputs=status_text_1)
            url_input.submit(fn, inputs=url_input, outputs=[slider1, shared_gallery, status_text_1])
            image.change(fn, inputs=image, outputs=[slider1, shared_gallery, status_text_1])

        with gr.Tab("Compose Lab", id="compose"):
            with gr.Column(elem_classes=["section-card"]):
                gr.HTML("""
                <div class="section-head">
                  <p>Scene Builder</p>
                  <h3>Place the subject into a new frame</h3>
                  <div class="subcopy">Load a cutout, drop in a background, then tune scale, alignment, and color until the scene feels coherent.</div>
                </div>
                """)
                with gr.Row(elem_classes=["media-stage"], equal_height=True):
                    with gr.Column(scale=4, elem_classes=["media-panel", "image-container"]):
                        selected_fg = gr.Image(type="pil", label="Foreground Subject", elem_classes=["image-custom"])
                    with gr.Column(scale=4, elem_classes=["media-panel", "image-container"]):
                        bg_image = gr.Image(type="pil", label="Background Plate", elem_classes=["image-custom"])
                    with gr.Column(scale=5, elem_classes=["preview-panel", "image-container"]):
                        preview_image = gr.Image(type="pil", label="Composite Preview", elem_classes=["image-custom"])

            with gr.Row(equal_height=True):
                with gr.Column(scale=7, elem_classes=["control-card"]):
                    with gr.Accordion("Placement Direction", open=True):
                        with gr.Row():
                            with gr.Column():
                                scale_slider = gr.Slider(
                                    minimum=1,
                                    maximum=200,
                                    value=100,
                                    label="Scale %",
                                    info="Resize the subject relative to the background"
                                )
                                rotation = gr.Slider(
                                    minimum=-180,
                                    maximum=180,
                                    value=0,
                                    step=1,
                                    label="Rotation",
                                    info="Rotate the subject in degrees"
                                )
                            with gr.Column():
                                x_offset = gr.Slider(
                                    minimum=-1000,
                                    maximum=1000,
                                    value=0,
                                    step=1,
                                    label="Horizontal Offset",
                                    info="Negative moves left, positive moves right"
                                )
                                y_offset = gr.Slider(
                                    minimum=-1000,
                                    maximum=1000,
                                    value=0,
                                    step=1,
                                    label="Vertical Offset",
                                    info="Positive values lift the subject upward"
                                )
                        with gr.Row(elem_classes=["control-actions"]):
                            flip_h = gr.Checkbox(
                                label="Mirror Horizontally",
                                value=False,
                                info="Flip the subject left to right"
                            )
                            flip_v = gr.Checkbox(
                                label="Mirror Vertically",
                                value=False,
                                info="Flip the subject top to bottom"
                            )
                        with gr.Row(elem_classes=["control-actions"]):
                            gr.Button("Reset Placement", size="sm").click(
                                reset_controls,
                                outputs=[scale_slider, x_offset, y_offset, rotation, flip_h, flip_v]
                            )
                            gr.Button("Fit Subject To BG", size="sm").click(
                                lambda fg, bg: calculate_fit_scale(fg, bg),
                                inputs=[selected_fg, bg_image],
                                outputs=scale_slider
                            )

                with gr.Column(scale=5, elem_classes=["control-card"]):
                    with gr.Accordion("Color Atmosphere", open=True):
                        brightness_slider = gr.Slider(
                            minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                            label="Brightness", info="Push overall lightness"
                        )
                        contrast_slider = gr.Slider(
                            minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                            label="Contrast", info="Increase or soften separation"
                        )
                        saturation_slider = gr.Slider(
                            minimum=0.0, maximum=2.0, value=1.0, step=0.05,
                            label="Saturation", info="Dial color intensity up or down"
                        )
                        temperature_slider = gr.Slider(
                            minimum=-50, maximum=50, value=0, step=1,
                            label="Temperature", info="Shift cooler or warmer"
                        )
                        tint_color = gr.ColorPicker(
                            label="Tint Color",
                            info="Overlay a color cast across the subject"
                        )
                        tint_strength = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                            label="Tint Strength", info="Control tint opacity"
                        )
                        with gr.Row(elem_classes=["control-actions"]):
                            reset_color_btn = gr.Button("Reset Colors", size="sm")
                            reset_color_btn.click(
                                reset_color_controls,
                                outputs=[brightness_slider, contrast_slider, saturation_slider,
                                        temperature_slider, tint_color, tint_strength]
                            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=4, elem_classes=["action-card"]):
                    save_btn = gr.Button("Save Composite", variant="primary", size="lg")
                    open_folder_btn_2 = gr.Button("Reveal Output Folder", size="lg")
                with gr.Column(scale=8, elem_classes=["status-card"]):
                    status_text_2 = gr.Textbox(label="Compose Status", interactive=False, lines=2)
                
            open_folder_btn_2.click(open_output_folder, outputs=status_text_2)
            save_btn.click(save_combined, inputs=[preview_image], outputs=[shared_gallery, status_text_2])
            
            color_controls = [
                brightness_slider, contrast_slider, saturation_slider,
                temperature_slider, tint_color, tint_strength
            ]
    
            all_controls = [
                selected_fg, bg_image, scale_slider, x_offset, y_offset,
                rotation, flip_h, flip_v, *color_controls
            ]
    
            for control in all_controls:
                control.change(
                    update_preview,
                    inputs=all_controls,
                    outputs=preview_image
                )
                
            selected_fg.change(
                handle_fg_change,
                inputs=[selected_fg, bg_image],
                outputs=[
                    preview_image,
                    scale_slider, x_offset, y_offset, rotation, flip_h, flip_v,
                    brightness_slider, contrast_slider, saturation_slider,
                    temperature_slider, tint_color, tint_strength
                ]
            )
            
        with gr.Tab("Batch Lab", id="batch"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=8, elem_classes=["section-card"]):
                    gr.HTML("""
                    <div class="section-head">
                      <p>Batch Queue</p>
                      <h3>Run a stack of images</h3>
                      <div class="subcopy">Load several assets at once, process them together, and keep original filenames with a `_nobg` suffix.</div>
                    </div>
                    """)
                    file_output = gr.File(
                        file_count="multiple",
                        label="Batch Input",
                        scale=2,
                    )
                    with gr.Row(equal_height=True, elem_classes=["action-row"]):
                        with gr.Column(scale=4, elem_classes=["action-card"]):
                            process_button = gr.Button("Process Batch", variant="primary", size="lg")
                            open_folder_btn_3 = gr.Button("Open Output Folder", size="lg")
                        with gr.Column(scale=8, elem_classes=["status-card"]):
                            status = gr.Textbox(label="Batch Status", lines=8)
                with gr.Column(scale=4, elem_classes=["info-card"]):
                    gr.HTML("""
                    <h3>Batch Workflow</h3>
                    <ul>
                      <li>Drag individual image files directly into the queue.</li>
                      <li>Use multi-select in the picker for larger sets.</li>
                      <li>Supported formats: JPG, PNG, WEBP, and GIF.</li>
                      <li>Folders are not supported by Gradio's file input.</li>
                    </ul>
                    """)
                    
            open_folder_btn_3.click(open_output_folder, outputs=status)
            process_button.click(batch_process_images, inputs=[file_output], outputs=[status, shared_gallery])

 
        
    # When a new foreground image is loaded
    selected_fg.change(
        lambda: (
            # Reset all controls to default values
            *reset_controls(),  # Returns (100, 0, 0, 0, False, False)
            *reset_color_controls(),  # Returns (1.0, 1.0, 1.0, 0, "#000000", 0)
        ),
        outputs=[
            scale_slider, x_offset, y_offset, rotation, flip_h, flip_v,
            brightness_slider, contrast_slider, saturation_slider,
            temperature_slider, tint_color, tint_strength
        ]
    )
        
demo.launch(server_name=HOST, server_port=PORT, share=False)
