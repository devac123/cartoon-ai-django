import time
import os
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
import io
# Using OpenCV and NumPy for all image processing (more reliable)


def advanced_cartoon_processing(image_path, output_path, style='classic'):
    """
    Advanced cartoon processing with multiple animation styles using OpenCV and scikit-image.
    Styles: 'classic', 'anime', 'sketch', 'watercolor', 'oil_painting'
    """
    try:
        processing_start = time.time()
        
        # Read image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if style == 'classic':
            result = classic_cartoon_effect(img)
        elif style == 'anime':
            result = anime_cartoon_effect(img)
        elif style == 'sketch':
            result = sketch_effect(img)
        elif style == 'watercolor':
            result = watercolor_effect(img)
        elif style == 'oil_painting':
            result = oil_painting_effect(img)
        elif style == 'neural':
            result = neural_cartoon_effect(img)
        else:
            result = neural_cartoon_effect(img)  # Default to neural for best results
        
        # Convert back to PIL Image for saving
        result_pil = Image.fromarray(result)
        result_pil.save(output_path, 'JPEG', quality=95)
        
        processing_time = time.time() - processing_start
        return True, processing_time, None
        
    except Exception as e:
        return False, 0, str(e)


def classic_cartoon_effect(img):
    """
    Creates a professional Disney-style cartoon effect with advanced processing
    """
    # Store original dimensions
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Resize for processing if too large (maintain aspect ratio)
    if max(width, height) > 1200:
        scale = 1200 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 1: Advanced noise reduction and surface smoothing
    # Use Non-local Means denoising for better quality
    smooth = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)
    
    # Progressive bilateral filtering for professional smoothing
    for d, sigma_color, sigma_space in [(7, 80, 80), (9, 120, 120), (11, 160, 160)]:
        smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)
    
    # Step 2: Professional edge detection with multiple scales
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale edge detection for better line art
    edges_multi = np.zeros_like(gray)
    
    # Fine details with adaptive threshold
    for block_size in [7, 11, 15]:
        edges_adaptive = cv2.adaptiveThreshold(
            cv2.GaussianBlur(gray, (3, 3), 0), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 8
        )
        edges_multi = cv2.bitwise_or(edges_multi, edges_adaptive)
    
    # Strong structural edges with Canny
    edges_canny = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    edges_multi = cv2.bitwise_or(edges_multi, edges_canny)
    
    # Morphological operations for cleaner edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    edges_multi = cv2.morphologyEx(edges_multi, cv2.MORPH_CLOSE, kernel)
    edges_multi = cv2.morphologyEx(edges_multi, cv2.MORPH_OPEN, np.ones((1,1), np.uint8))
    
    # Convert to 3-channel
    edges = cv2.cvtColor(edges_multi, cv2.COLOR_GRAY2RGB)
    
    # Step 3: Intelligent color quantization with perceptual color space
    # Convert to LAB for better perceptual uniformity
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    data = lab.reshape((-1, 3))
    data = np.float32(data)
    
    # Adaptive clustering based on image complexity
    unique_colors = len(np.unique(data.view(np.void), axis=0))
    k = min(max(10, unique_colors // 5000), 18)  # Adaptive cluster count
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 15, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    
    quantized_lab = centers[labels.flatten()]
    quantized_lab = quantized_lab.reshape(lab.shape)
    quantized = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2RGB)
    
    # Step 4: Professional color enhancement
    # Enhance saturation and vibrancy in HSV space
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Selective saturation boost (stronger for less saturated colors)
    saturation = hsv[:, :, 1] / 255.0
    saturation_boost = 1.0 + (1.0 - saturation) * 0.6  # Adaptive boost
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    
    # Brightness enhancement with gamma correction
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
    
    quantized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Step 5: Advanced edge integration with soft blending
    edges_inv = cv2.bitwise_not(edges)
    edges_soft = cv2.GaussianBlur(edges_inv, (3, 3), 0) / 255.0
    
    # Soft edge blending
    cartoon = quantized.astype(np.float32)
    for i in range(3):
        cartoon[:, :, i] = cartoon[:, :, i] * edges_soft[:, :, i]
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Step 6: Professional depth and lighting effects
    gray_cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2GRAY)
    
    # Create sophisticated shadow and highlight masks
    shadows = cv2.threshold(gray_cartoon, 80, 255, cv2.THRESH_BINARY_INV)[1]
    shadows = cv2.GaussianBlur(shadows, (7, 7), 0)
    
    highlights = cv2.threshold(gray_cartoon, 180, 255, cv2.THRESH_BINARY)[1]
    highlights = cv2.GaussianBlur(highlights, (7, 7), 0)
    
    # Apply depth effects
    cartoon = cartoon.astype(np.float32)
    shadow_effect = (shadows / 255.0) * 0.15
    highlight_effect = (highlights / 255.0) * 0.2
    
    for i in range(3):
        cartoon[:, :, i] = cartoon[:, :, i] * (1 - shadow_effect) + highlight_effect * 255
    
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Step 7: Final professional polish
    # Subtle sharpening for crisp details
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * 0.1
    cartoon = cv2.filter2D(cartoon, -1, kernel_sharpen)
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Final bilateral filter for smooth finish
    cartoon = cv2.bilateralFilter(cartoon, 5, 40, 40)
    
    # Resize back to original dimensions with high-quality interpolation
    if original_size != (cartoon.shape[1], cartoon.shape[0]):
        cartoon = cv2.resize(cartoon, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    return cartoon


def anime_cartoon_effect(img):
    """
    Creates a professional anime-style cartoon effect with clean cel-shading
    """
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Resize for processing if needed
    if max(width, height) > 1000:
        scale = 1000 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 1: Advanced denoising and super-smooth regions (anime characteristic)
    smooth = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Multiple bilateral filters for ultra-smooth anime regions
    for d, sigma_color, sigma_space in [(9, 100, 100), (15, 150, 150), (20, 200, 200)]:
        smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)
    
    # Step 2: Professional anime-style edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Create clean, thick edges typical of anime
    edges_thick = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (7, 7), 0), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8
    )
    
    # Add fine detail edges
    edges_fine = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 80)
    
    # Combine and clean edges
    edges_combined = cv2.bitwise_or(edges_thick, edges_fine)
    
    # Morphological operations for cleaner anime lines
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_combined = cv2.morphologyEx(edges_combined, cv2.MORPH_CLOSE, kernel_close)
    
    # Dilate slightly for thicker anime-style lines
    kernel_dilate = np.ones((2,2), np.uint8)
    edges_combined = cv2.dilate(edges_combined, kernel_dilate, iterations=1)
    
    edges = cv2.cvtColor(edges_combined, cv2.COLOR_GRAY2RGB)
    
    # Step 3: Aggressive color quantization for flat anime colors
    # Convert to LAB for better color clustering
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    data = lab.reshape((-1, 3))
    data = np.float32(data)
    
    # Use fewer clusters for typical anime flat coloring
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, 8, None, criteria, 15, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    
    quantized_lab = centers[labels.flatten()]
    quantized_lab = quantized_lab.reshape(lab.shape)
    quantized = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2RGB)
    
    # Step 4: Anime-style color enhancement
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Strong saturation boost for vibrant anime colors
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.6, 0, 255)
    
    # Brightness adjustment for anime look
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
    
    quantized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Step 5: Create cel-shading effect (anime characteristic)
    gray_quant = cv2.cvtColor(quantized, cv2.COLOR_RGB2GRAY)
    
    # Create multiple threshold levels for cel-shading
    levels = [60, 120, 180]
    cel_shaded = quantized.copy().astype(np.float32)
    
    for level in levels:
        mask = (gray_quant > level).astype(np.float32)
        for i in range(3):
            cel_shaded[:, :, i] = cel_shaded[:, :, i] * (0.9 + 0.1 * mask)
    
    cel_shaded = np.clip(cel_shaded, 0, 255).astype(np.uint8)
    
    # Step 6: Combine with edges using soft blending
    edges_inv = cv2.bitwise_not(edges)
    edges_soft = cv2.GaussianBlur(edges_inv, (2, 2), 0) / 255.0
    
    cartoon = cel_shaded.astype(np.float32)
    for i in range(3):
        cartoon[:, :, i] = cartoon[:, :, i] * edges_soft[:, :, i]
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Step 7: Final anime polish
    # Slight blur to soften harsh transitions
    cartoon = cv2.GaussianBlur(cartoon, (1, 1), 0)
    
    # Resize back to original dimensions
    if original_size != (cartoon.shape[1], cartoon.shape[0]):
        cartoon = cv2.resize(cartoon, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    return cartoon


def sketch_effect(img):
    """
    Creates a professional pencil sketch effect with artistic shading
    """
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Resize for processing if needed
    if max(width, height) > 1000:
        scale = 1000 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 1: Advanced grayscale conversion with weighted channels
    # Use luminosity method for better contrast
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply subtle denoising to clean up the sketch
    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # Step 2: Create multiple sketch layers for depth
    sketches = []
    
    # Fine detail sketch
    inverted_fine = 255 - gray
    blurred_fine = cv2.GaussianBlur(inverted_fine, (21, 21), 0)
    sketch_fine = cv2.divide(gray, 255 - blurred_fine, scale=256)
    sketches.append(sketch_fine)
    
    # Medium detail sketch
    blurred_medium = cv2.GaussianBlur(inverted_fine, (45, 45), 0)
    sketch_medium = cv2.divide(gray, 255 - blurred_medium, scale=256)
    sketches.append(sketch_medium)
    
    # Coarse shading sketch
    blurred_coarse = cv2.GaussianBlur(inverted_fine, (81, 81), 0)
    sketch_coarse = cv2.divide(gray, 255 - blurred_coarse, scale=256)
    sketches.append(sketch_coarse)
    
    # Step 3: Combine sketches with different weights
    final_sketch = np.zeros_like(gray, dtype=np.float32)
    weights = [0.5, 0.3, 0.2]  # Fine, medium, coarse
    
    for sketch, weight in zip(sketches, weights):
        final_sketch += sketch.astype(np.float32) * weight
    
    final_sketch = np.clip(final_sketch, 0, 255).astype(np.uint8)
    
    # Step 4: Add artistic texture and enhance contrast
    # Apply adaptive histogram equalization for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    final_sketch = clahe.apply(final_sketch)
    
    # Add subtle paper texture effect
    rows, cols = final_sketch.shape
    noise = np.random.normal(0, 8, (rows, cols))
    final_sketch = final_sketch.astype(np.float32) + noise
    final_sketch = np.clip(final_sketch, 0, 255).astype(np.uint8)
    
    # Step 5: Enhance line definition with edge enhancement
    # Create edge mask to preserve important lines
    edges = cv2.Canny(gray, 50, 150)
    edges_dilated = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    
    # Darken edges in the sketch
    edge_mask = edges_dilated.astype(np.float32) / 255.0
    final_sketch = final_sketch.astype(np.float32)
    final_sketch = final_sketch * (1 - edge_mask * 0.3)  # Darken edges
    final_sketch = np.clip(final_sketch, 0, 255).astype(np.uint8)
    
    # Step 6: Add subtle color tinting for warmth
    sketch_colored = cv2.cvtColor(final_sketch, cv2.COLOR_GRAY2RGB)
    
    # Add warm sepia-like tint
    sketch_colored = sketch_colored.astype(np.float32)
    sketch_colored[:, :, 0] = np.clip(sketch_colored[:, :, 0] * 1.05, 0, 255)  # Slight red
    sketch_colored[:, :, 1] = np.clip(sketch_colored[:, :, 1] * 1.02, 0, 255)  # Slight green
    sketch_colored[:, :, 2] = np.clip(sketch_colored[:, :, 2] * 0.95, 0, 255)  # Reduce blue
    sketch_colored = sketch_colored.astype(np.uint8)
    
    # Resize back to original dimensions
    if original_size != (sketch_colored.shape[1], sketch_colored.shape[0]):
        sketch_colored = cv2.resize(sketch_colored, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    return sketch_colored


def watercolor_effect(img):
    """
    Creates a professional watercolor painting effect with artistic flow and transparency
    """
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Resize for processing if needed
    if max(width, height) > 1000:
        scale = 1000 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 1: Create watercolor base with multiple layers
    # Heavy denoising for clean watercolor regions
    smooth = cv2.fastNlMeansDenoisingColored(img, None, 15, 15, 7, 21)
    
    # Multiple bilateral filters for ultra-smooth watercolor regions
    watercolor_base = smooth.copy()
    for d, sigma_color, sigma_space in [(12, 100, 100), (20, 200, 200), (25, 300, 300)]:
        watercolor_base = cv2.bilateralFilter(watercolor_base, d, sigma_color, sigma_space)
    
    # Step 2: Create watercolor flow effects
    # Apply directional blur to simulate paint flow
    kernel_horizontal = np.ones((1, 15), np.float32) / 15
    kernel_vertical = np.ones((15, 1), np.float32) / 15
    kernel_diagonal1 = np.eye(15, dtype=np.float32) / 15
    kernel_diagonal2 = np.flip(np.eye(15, dtype=np.float32), axis=1) / 15
    
    # Apply different directional flows
    flow_h = cv2.filter2D(watercolor_base, -1, kernel_horizontal)
    flow_v = cv2.filter2D(watercolor_base, -1, kernel_vertical)
    flow_d1 = cv2.filter2D(watercolor_base, -1, kernel_diagonal1)
    flow_d2 = cv2.filter2D(watercolor_base, -1, kernel_diagonal2)
    
    # Combine flows with weights
    watercolor_flow = (flow_h * 0.3 + flow_v * 0.3 + flow_d1 * 0.2 + flow_d2 * 0.2).astype(np.uint8)
    
    # Step 3: Create artistic color bleeding effect
    # Convert to LAB for better color manipulation
    lab = cv2.cvtColor(watercolor_flow, cv2.COLOR_RGB2LAB)
    
    # Apply morphological operations to create color bleeding
    kernel_bleed = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    lab[:, :, 1] = cv2.morphologyEx(lab[:, :, 1], cv2.MORPH_CLOSE, kernel_bleed)  # A channel
    lab[:, :, 2] = cv2.morphologyEx(lab[:, :, 2], cv2.MORPH_CLOSE, kernel_bleed)  # B channel
    
    watercolor_bled = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Step 4: Create watercolor transparency effects
    # Simulate transparent watercolor layers
    transparency_mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    transparency_mask = cv2.GaussianBlur(transparency_mask, (15, 15), 0)
    transparency_mask = transparency_mask.astype(np.float32) / 255.0
    
    # Create multiple transparency layers
    layer1 = watercolor_bled.astype(np.float32)
    layer2 = cv2.GaussianBlur(watercolor_bled, (21, 21), 0).astype(np.float32)
    layer3 = cv2.medianBlur(watercolor_bled, 25).astype(np.float32)
    
    # Blend layers with transparency
    watercolor_transparent = layer1 * 0.6 + layer2 * 0.25 + layer3 * 0.15
    watercolor_transparent = np.clip(watercolor_transparent, 0, 255).astype(np.uint8)
    
    # Step 5: Enhance watercolor colors
    hsv = cv2.cvtColor(watercolor_transparent, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Boost saturation for vibrant watercolor effect
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    
    # Adjust brightness for watercolor luminosity
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    
    watercolor_enhanced = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Step 6: Add subtle paper texture
    # Create paper-like texture
    rows, cols, _ = watercolor_enhanced.shape
    paper_texture = np.random.normal(0, 5, (rows, cols))
    paper_texture = cv2.GaussianBlur(paper_texture, (3, 3), 0)
    
    # Apply texture to all channels
    watercolor_textured = watercolor_enhanced.astype(np.float32)
    for i in range(3):
        watercolor_textured[:, :, i] += paper_texture * 0.3
    
    watercolor_textured = np.clip(watercolor_textured, 0, 255).astype(np.uint8)
    
    # Step 7: Create soft watercolor edges
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Very soft edge detection for watercolor
    edges_soft = cv2.Canny(cv2.GaussianBlur(gray, (7, 7), 0), 20, 60)
    
    # Heavily blur edges for watercolor softness
    edges_soft = cv2.GaussianBlur(edges_soft, (9, 9), 0)
    edges_soft = cv2.morphologyEx(edges_soft, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    
    # Create soft edge mask
    edge_mask = edges_soft.astype(np.float32) / 255.0
    edge_mask_3d = np.stack([edge_mask] * 3, axis=2)
    
    # Apply soft edge darkening
    watercolor_final = watercolor_textured.astype(np.float32)
    watercolor_final = watercolor_final * (1 - edge_mask_3d * 0.2)
    watercolor_final = np.clip(watercolor_final, 0, 255).astype(np.uint8)
    
    # Step 8: Final watercolor polish
    # Add slight color variation for artistic effect
    watercolor_final = watercolor_final.astype(np.float32)
    
    # Subtle color temperature variation
    watercolor_final[:, :, 0] = np.clip(watercolor_final[:, :, 0] * 1.03, 0, 255)  # Warm reds
    watercolor_final[:, :, 2] = np.clip(watercolor_final[:, :, 2] * 0.97, 0, 255)  # Cool blues
    
    watercolor_final = watercolor_final.astype(np.uint8)
    
    # Final soft blur for watercolor finish
    watercolor_final = cv2.GaussianBlur(watercolor_final, (2, 2), 0)
    
    # Resize back to original dimensions
    if original_size != (watercolor_final.shape[1], watercolor_final.shape[0]):
        watercolor_final = cv2.resize(watercolor_final, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    return watercolor_final


def oil_painting_effect(img):
    """
    Creates an oil painting effect
    """
    # Apply strong bilateral filter
    bilateral = cv2.bilateralFilter(img, 20, 60, 60)
    
    # Apply additional smoothing
    oil_painting = cv2.bilateralFilter(bilateral, 15, 80, 80)
    
    # Enhance colors
    hsv = cv2.cvtColor(oil_painting, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)  # Increase saturation
    hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.1)  # Increase brightness slightly
    oil_painting = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return oil_painting


def neural_cartoon_effect(img):
    """
    Creates a state-of-the-art cartoon effect using advanced computer vision techniques
    mimicking neural style transfer results
    """
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Resize for processing with better quality
    if max(width, height) > 1400:
        scale = 1400 / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Step 1: Ultra-advanced denoising and surface preparation
    # Apply Non-local Means denoising with optimized parameters
    smooth = cv2.fastNlMeansDenoisingColored(img, None, 12, 12, 7, 21)
    
    # Progressive multi-scale bilateral filtering for professional smoothing
    smoothing_stages = [
        (5, 60, 60),   # Fine details preservation
        (9, 120, 120), # Medium smoothing
        (13, 180, 180), # Strong smoothing
        (17, 240, 240)  # Ultra-smooth cartoon regions
    ]
    
    for d, sigma_color, sigma_space in smoothing_stages:
        smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)
    
    # Step 2: Intelligent face and skin detection for enhanced processing
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Skin color detection in HSV space for better skin smoothing
    hsv = cv2.cvtColor(smooth, cv2.COLOR_RGB2HSV)
    
    # Define skin color range in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Morphological operations to clean skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    
    # Apply extra smoothing to skin regions
    skin_regions = cv2.bitwise_and(smooth, smooth, mask=skin_mask)
    skin_smooth = cv2.bilateralFilter(skin_regions, 21, 300, 300)
    
    # Blend smoothed skin back
    skin_mask_3d = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2RGB) / 255.0
    smooth = smooth.astype(np.float32)
    skin_smooth = skin_smooth.astype(np.float32)
    smooth = smooth * (1 - skin_mask_3d) + skin_smooth * skin_mask_3d
    smooth = np.clip(smooth, 0, 255).astype(np.uint8)
    
    # Step 3: Professional multi-scale edge detection
    edges_pyramid = []
    
    # Create Gaussian pyramid for multi-scale edge detection
    scales = [1.0, 0.8, 0.6, 0.4]
    
    for scale in scales:
        if scale != 1.0:
            scaled_img = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            scaled_img = gray.copy()
        
        # Multiple edge detection methods
        # 1. Adaptive threshold edges
        edges_adaptive = cv2.adaptiveThreshold(
            cv2.GaussianBlur(scaled_img, (5, 5), 0), 255,
            cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 10
        )
        
        # 2. Canny edges with multiple thresholds
        edges_canny_low = cv2.Canny(cv2.GaussianBlur(scaled_img, (3, 3), 0), 30, 90)
        edges_canny_high = cv2.Canny(cv2.GaussianBlur(scaled_img, (3, 3), 0), 50, 150)
        
        # Combine edge methods
        edges_combined = cv2.bitwise_or(edges_adaptive, edges_canny_low)
        edges_combined = cv2.bitwise_or(edges_combined, edges_canny_high)
        
        # Resize back to original scale
        if scale != 1.0:
            edges_combined = cv2.resize(edges_combined, (gray.shape[1], gray.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)
        
        edges_pyramid.append(edges_combined)
    
    # Combine multi-scale edges
    edges_final = np.zeros_like(gray)
    for edges in edges_pyramid:
        edges_final = cv2.bitwise_or(edges_final, edges)
    
    # Advanced morphological operations for cleaner edges
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    
    edges_final = cv2.morphologyEx(edges_final, cv2.MORPH_CLOSE, kernel_close)
    edges_final = cv2.morphologyEx(edges_final, cv2.MORPH_OPEN, kernel_open)
    
    # Convert to 3-channel
    edges_3d = cv2.cvtColor(edges_final, cv2.COLOR_GRAY2RGB)
    
    # Step 4: Advanced perceptual color quantization
    # Convert to perceptually uniform LAB color space
    lab = cv2.cvtColor(smooth, cv2.COLOR_RGB2LAB)
    data = lab.reshape((-1, 3))
    data = np.float32(data)
    
    # Intelligent adaptive clustering
    unique_colors = len(np.unique(data.view(np.void), axis=0))
    # More sophisticated cluster count calculation
    k = min(max(12, int(np.sqrt(unique_colors / 1000))), 20)
    
    # Enhanced K-means with better initialization
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 20, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    
    quantized_lab = centers[labels.flatten()]
    quantized_lab = quantized_lab.reshape(lab.shape)
    quantized = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2RGB)
    
    # Step 5: Professional color enhancement pipeline
    # Convert to HSV for color manipulation
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV).astype(np.float32)
    
    # Advanced saturation enhancement (preserve natural skin tones)
    saturation = hsv[:, :, 1] / 255.0
    
    # Create saturation boost map (less boost for skin-like colors)
    hue = hsv[:, :, 0]
    skin_hue_mask = ((hue >= 0) & (hue <= 25)) | ((hue >= 160) & (hue <= 180))
    
    saturation_boost = np.where(skin_hue_mask, 1.2, 1.5)  # Less boost for skin tones
    saturation_boost *= (1.0 + (1.0 - saturation) * 0.5)   # Adaptive boost
    
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255)
    
    # Advanced brightness enhancement with gamma correction
    value = hsv[:, :, 2] / 255.0
    gamma = 0.8  # Gamma correction for better contrast
    value_enhanced = np.power(value, gamma) * 255.0
    hsv[:, :, 2] = np.clip(value_enhanced * 1.1, 0, 255)
    
    quantized = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    # Step 6: Advanced edge integration with soft blending
    edges_inv = cv2.bitwise_not(edges_3d)
    
    # Create soft edge mask for gradual blending
    edges_soft = cv2.GaussianBlur(edges_inv, (3, 3), 0).astype(np.float32) / 255.0
    
    # Apply soft edge blending
    cartoon = quantized.astype(np.float32)
    for i in range(3):
        cartoon[:, :, i] = cartoon[:, :, i] * edges_soft[:, :, i]
    cartoon = np.clip(cartoon, 0, 255).astype(np.uint8)
    
    # Step 7: Professional depth and lighting simulation
    gray_cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2GRAY)
    
    # Create sophisticated shadow and highlight masks
    # Multiple threshold levels for better depth
    shadow_levels = [50, 80, 110]
    highlight_levels = [170, 200, 230]
    
    depth_effect = cartoon.astype(np.float32)
    
    # Apply shadow effects
    for level in shadow_levels:
        shadow_mask = cv2.threshold(gray_cartoon, level, 255, cv2.THRESH_BINARY_INV)[1]
        shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0).astype(np.float32) / 255.0
        shadow_strength = (shadow_levels.index(level) + 1) * 0.05
        
        for i in range(3):
            depth_effect[:, :, i] = depth_effect[:, :, i] * (1 - shadow_mask * shadow_strength)
    
    # Apply highlight effects
    for level in highlight_levels:
        highlight_mask = cv2.threshold(gray_cartoon, level, 255, cv2.THRESH_BINARY)[1]
        highlight_mask = cv2.GaussianBlur(highlight_mask, (7, 7), 0).astype(np.float32) / 255.0
        highlight_strength = (highlight_levels.index(level) + 1) * 0.08
        
        for i in range(3):
            depth_effect[:, :, i] = depth_effect[:, :, i] + highlight_mask * highlight_strength * 255
    
    cartoon = np.clip(depth_effect, 0, 255).astype(np.uint8)
    
    # Step 8: Final professional polish and sharpening
    # Unsharp masking for crisp details
    gaussian = cv2.GaussianBlur(cartoon, (3, 3), 0)
    unsharp_mask = cv2.addWeighted(cartoon, 1.5, gaussian, -0.5, 0)
    cartoon = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
    
    # Final bilateral filter for smooth finish while preserving edges
    cartoon = cv2.bilateralFilter(cartoon, 5, 50, 50)
    
    # Subtle color temperature adjustment for warmer cartoon look
    cartoon = cartoon.astype(np.float32)
    cartoon[:, :, 0] = np.clip(cartoon[:, :, 0] * 1.02, 0, 255)  # Slight red boost
    cartoon[:, :, 1] = np.clip(cartoon[:, :, 1] * 1.01, 0, 255)  # Slight green boost
    cartoon = cartoon.astype(np.uint8)
    
    # Resize back to original dimensions with high-quality interpolation
    if original_size != (cartoon.shape[1], cartoon.shape[0]):
        cartoon = cv2.resize(cartoon, original_size, interpolation=cv2.INTER_LANCZOS4)
    
    return cartoon


def process_image_to_cartoon(image_processing_obj):
    """
    Process an ImageProcessing object to create cartoon version using neural style (default)
    """
    return process_image_with_style(image_processing_obj, 'neural')


def process_image_with_style(image_processing_obj, style='neural'):
    """
    Process an ImageProcessing object to create cartoon version with specified style
    """
    try:
        # Update status to processing
        image_processing_obj.status = 'processing'
        image_processing_obj.save()
        
        # Get the original image path
        original_path = image_processing_obj.original_image.path
        
        # Create output filename
        original_name = os.path.basename(original_path)
        name, ext = os.path.splitext(original_name)
        output_filename = f"{name}_cartoon_{style}{ext}"
        
        # Create temporary output path
        temp_output_path = os.path.join(
            os.path.dirname(original_path), 
            f"temp_{output_filename}"
        )
        
        # Process the image with the specified style
        success, processing_time, error_msg = advanced_cartoon_processing(
            original_path, 
            temp_output_path,
            style=style
        )
        
        if success:
            # Save the processed image to the model
            with open(temp_output_path, 'rb') as f:
                image_processing_obj.processed_image.save(
                    output_filename,
                    ContentFile(f.read()),
                    save=False
                )
            
            # Update status and processing time
            image_processing_obj.status = 'completed'
            image_processing_obj.processing_time = processing_time
            image_processing_obj.save()
            
            # Clean up temporary file
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
            
            return True, None
        else:
            # Update status to failed
            image_processing_obj.status = 'failed'
            image_processing_obj.error_message = error_msg
            image_processing_obj.save()
            
            return False, error_msg
            
    except Exception as e:
        # Update status to failed
        image_processing_obj.status = 'failed'
        image_processing_obj.error_message = str(e)
        image_processing_obj.save()
        
        return False, str(e)


# Future integration functions for real AI models
def integrate_toonify_api(image_path, api_key):
    """
    Placeholder for Toonify API integration
    """
    # TODO: Implement actual Toonify API calls
    pass


def integrate_animegan_model(image_path, model_path):
    """
    Placeholder for AnimeGAN model integration
    """
    # TODO: Implement actual AnimeGAN model processing
    pass
