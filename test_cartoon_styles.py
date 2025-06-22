#!/usr/bin/env python
"""
Demo script to test different cartoon styles
Run this script to test the cartoon processing functions
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cartoon_project.settings')
django.setup()

from image_processor.ai_utils import advanced_cartoon_processing
import cv2
import numpy as np

def create_test_image():
    """Create a simple test image if no image is provided"""
    # Create a colorful test image with gradients and shapes
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Background gradient
    for i in range(400):
        img[i, :] = [int(255 * i / 400), int(100 + 155 * i / 400), 200]
    
    # Add some shapes
    cv2.circle(img, (150, 150), 50, (255, 100, 100), -1)
    cv2.rectangle(img, (250, 100), (350, 200), (100, 255, 100), -1)
    cv2.ellipse(img, (450, 150), (60, 40), 0, 0, 360, (100, 100, 255), -1)
    
    # Save test image
    test_path = 'media/test_input.jpg'
    os.makedirs('media', exist_ok=True)
    cv2.imwrite(test_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return test_path

def test_all_styles():
    """Test all available cartoon styles"""
    
    # Create or use existing test image
    test_image_path = create_test_image()
    
    styles = ['neural', 'classic', 'anime', 'sketch', 'watercolor', 'oil_painting']
    
    print("🎨 Testing Cartoon AI Styles...")
    print("=" * 50)
    
    for style in styles:
        print(f"\n🔄 Processing with '{style}' style...")
        
        output_path = f'media/test_output_{style}.jpg'
        
        try:
            success, processing_time, error_msg = advanced_cartoon_processing(
                test_image_path, 
                output_path, 
                style=style
            )
            
            if success:
                print(f"✅ {style.title()} style completed in {processing_time:.2f}s")
                print(f"   Output saved: {output_path}")
            else:
                print(f"❌ {style.title()} style failed: {error_msg}")
                
        except Exception as e:
            print(f"❌ {style.title()} style error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 Style testing completed!")
    print("\nRecommendation:")
    print("- 'neural' style: Best overall quality (default)")
    print("- 'classic': Disney-like cartoon effect")
    print("- 'anime': Japanese animation style")
    print("- 'sketch': Pencil drawing effect")
    print("- 'watercolor': Painting effect")
    print("- 'oil_painting': Oil painting effect")

if __name__ == '__main__':
    test_all_styles()
