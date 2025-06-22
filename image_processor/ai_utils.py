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
    Creates a high-quality Disney-style cartoon effect with advanced processing
    """
    # Store original dimensions
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Resize for processing if too large
    if width > 1000:
        scale = 1000 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Step 1: Multiple bilateral filters for ultra-smooth skin/surfaces
    smooth = img.copy()
    for _ in range(3):
        smooth = cv2.bilateralFilter(smooth, 9, 200, 200)
    
    # Step 2: Create detailed edge mask with multiple techniques
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Line art edges
    edges1 = cv2.adaptiveThreshold(cv2.medianBlur(gray, 7), 255, 
                                   cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 7)
    
    # Canny edges for fine details
    edges2 = cv2.Canny(gray, 50, 150)
    
    # Combine edge masks
    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Step 3: Advanced color quantization with skin tone preservation
    data = smooth.reshape((-1, 3))
    data = np.float32(data)
    
    # Use more clusters for better color preservation
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(data, 12, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(smooth.shape)
    
    # Step 4: Enhance colors for cartoon look
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.2)  # Boost saturation
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 10)        # Slight brightness boost
    quantized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Step 5: Advanced edge blending
    edges_inv = cv2.bitwise_not(edges)
    cartoon = cv2.bitwise_and(quantized, edges_inv)
    
    # Step 6: Add subtle texture and depth
    # Create highlight mask
    gray_cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2GRAY)
    highlights = cv2.threshold(gray_cartoon, 200, 255, cv2.THRESH_BINARY)[1]
    highlights = cv2.cvtColor(highlights, cv2.COLOR_GRAY2RGB)
    
    # Brighten highlights slightly
    cartoon = cv2.addWeighted(cartoon, 0.9, highlights, 0.1, 0)
    
    # Step 7: Final smoothing pass
    cartoon = cv2.bilateralFilter(cartoon, 5, 50, 50)
    
    # Resize back to original dimensions
    if original_size[0] > 1000:
        cartoon = cv2.resize(cartoon, original_size, interpolation=cv2.INTER_CUBIC)
    
    return cartoon


def anime_cartoon_effect(img):
    """
    Creates an anime-style cartoon effect
    """
    # Step 1: Apply strong bilateral filter for smooth regions
    bilateral = cv2.bilateralFilter(img, 20, 50, 50)
    
    # Step 2: Create edge mask
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 9, 9)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Step 3: More aggressive color quantization for anime look
    data = bilateral.reshape((-1, 3))
    data = np.float32(data)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(data, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(bilateral.shape)
    
    # Step 4: Enhance saturation for vibrant anime colors
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)  # Increase saturation
    quantized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Step 5: Combine with edges
    cartoon = cv2.bitwise_and(quantized, edges)
    
    return cartoon


def sketch_effect(img):
    """
    Creates a pencil sketch effect
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Create inverted image
    inverted = 255 - gray
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(inverted, (25, 25), 0)
    
    # Create sketch by dividing
    def dodge(x, y):
        return cv2.divide(x, 255 - y, scale=256)
    
    sketch = dodge(gray, blurred)
    
    # Convert back to 3-channel
    sketch_colored = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
    
    return sketch_colored


def watercolor_effect(img):
    """
    Creates a watercolor painting effect
    """
    # Apply bilateral filter for smooth areas
    bilateral = cv2.bilateralFilter(img, 10, 40, 40)
    
    # Apply median filter to create watercolor-like texture
    watercolor = cv2.medianBlur(bilateral, 19)
    
    # Create soft edges
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                  cv2.THRESH_BINARY, 7, 7)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Blend edges softly
    edges = cv2.GaussianBlur(edges, (3, 3), 0)
    
    # Combine
    result = cv2.bitwise_and(watercolor, edges)
    
    return result


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
    Creates a more advanced cartoon effect using neural-style techniques
    """
    height, width = img.shape[:2]
    original_size = (width, height)
    
    # Resize for processing
    if width > 800:
        scale = 800 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Step 1: Advanced denoising and smoothing
    smooth = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # Multiple bilateral filters with different parameters
    for d, sigma_color, sigma_space in [(5, 50, 50), (7, 100, 100), (9, 150, 150)]:
        smooth = cv2.bilateralFilter(smooth, d, sigma_color, sigma_space)
    
    # Step 2: Face-aware processing (if faces detected)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Load face cascade (built into OpenCV)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Apply special processing to face regions
        for (x, y, w, h) in faces:
            face_region = smooth[y:y+h, x:x+w]
            # Extra smoothing for faces
            face_region = cv2.bilateralFilter(face_region, 15, 200, 200)
            smooth[y:y+h, x:x+w] = face_region
    except:
        pass  # Continue without face detection if cascade not available
    
    # Step 3: Advanced edge detection with multiple scales
    edges_final = np.zeros(gray.shape, dtype=np.uint8)
    
    # Multi-scale edge detection
    for ksize in [3, 5, 7]:
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                      cv2.THRESH_BINARY, 7, 7)
        edges_final = cv2.bitwise_or(edges_final, edges)
    
    # Refine edges
    kernel = np.ones((2,2), np.uint8)
    edges_final = cv2.morphologyEx(edges_final, cv2.MORPH_CLOSE, kernel)
    edges_final = cv2.morphologyEx(edges_final, cv2.MORPH_OPEN, kernel)
    
    # Convert to 3-channel
    edges_3d = cv2.cvtColor(edges_final, cv2.COLOR_GRAY2RGB)
    
    # Step 4: Intelligent color quantization
    data = smooth.reshape((-1, 3))
    data = np.float32(data)
    
    # Adaptive K-means based on image complexity
    unique_colors = len(np.unique(data.view(np.void), axis=0))
    k = min(max(8, unique_colors // 10000), 16)  # Adaptive cluster count
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(smooth.shape)
    
    # Step 5: Color enhancement with histogram equalization
    lab = cv2.cvtColor(quantized, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:, :, 0])
    quantized = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Boost saturation and adjust brightness
    hsv = cv2.cvtColor(quantized, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.3)  # Saturation boost
    hsv[:, :, 2] = cv2.add(hsv[:, :, 2], 15)        # Brightness boost
    quantized = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Step 6: Advanced blending
    edges_inv = cv2.bitwise_not(edges_3d)
    cartoon = cv2.bitwise_and(quantized, edges_inv)
    
    # Add depth with shadows and highlights
    gray_cartoon = cv2.cvtColor(cartoon, cv2.COLOR_RGB2GRAY)
    
    # Create shadow mask
    shadows = cv2.threshold(gray_cartoon, 60, 255, cv2.THRESH_BINARY_INV)[1]
    shadows = cv2.GaussianBlur(shadows, (5, 5), 0)
    shadows_3d = cv2.cvtColor(shadows, cv2.COLOR_GRAY2RGB)
    
    # Create highlight mask
    highlights = cv2.threshold(gray_cartoon, 180, 255, cv2.THRESH_BINARY)[1]
    highlights = cv2.GaussianBlur(highlights, (5, 5), 0)
    highlights_3d = cv2.cvtColor(highlights, cv2.COLOR_GRAY2RGB)
    
    # Apply shadows and highlights
    cartoon = cv2.addWeighted(cartoon, 0.85, shadows_3d, -0.1, 0)  # Darken shadows
    cartoon = cv2.addWeighted(cartoon, 0.9, highlights_3d, 0.1, 0)  # Brighten highlights
    
    # Final polish
    cartoon = cv2.bilateralFilter(cartoon, 3, 30, 30)
    
    # Resize back to original
    if original_size[0] > 800:
        cartoon = cv2.resize(cartoon, original_size, interpolation=cv2.INTER_CUBIC)
    
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
