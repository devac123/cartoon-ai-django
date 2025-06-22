#!/usr/bin/env python
"""
Test script to process the latest uploaded image
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cartoon_project.settings')
django.setup()

from image_processor.models import ImageProcessing
from image_processor.ai_utils import process_image_to_cartoon, advanced_cartoon_processing
import cv2
import numpy as np

def test_opencv():
    """Test if OpenCV is working properly"""
    print("🔬 Testing OpenCV functionality...")
    
    try:
        # Create a simple test image
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        test_img[25:75, 25:75] = [255, 255, 255]  # White square
        
        # Test basic OpenCV operations
        gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 50, 150)
        
        print("✅ OpenCV basic operations working")
        return True
    except Exception as e:
        print(f"❌ OpenCV test failed: {str(e)}")
        return False

def process_all_pending():
    """Process all pending images"""
    print("🔄 Processing all pending images...")
    
    # Get all images that are not completed
    pending_images = ImageProcessing.objects.exclude(status='completed').order_by('-created_at')
    
    if not pending_images:
        print("📭 No pending images found")
        
        # Show latest 3 images regardless of status
        latest_images = ImageProcessing.objects.all().order_by('-created_at')[:3]
        if latest_images:
            print("\n📋 Latest images (any status):")
            for img in latest_images:
                print(f"  ID: {img.id}, Status: {img.status}, Created: {img.created_at}")
        return
    
    for img in pending_images:
        print(f"\n📝 Processing image ID: {img.id}")
        print(f"   Status: {img.status}")
        print(f"   File: {img.original_image.name}")
        
        if not os.path.exists(img.original_image.path):
            print(f"   ❌ File not found: {img.original_image.path}")
            img.status = 'failed'
            img.error_message = 'Original file not found'
            img.save()
            continue
        
        try:
            print("   🔄 Starting processing...")
            success, error = process_image_to_cartoon(img)
            
            if success:
                print("   ✅ Processing completed successfully!")
            else:
                print(f"   ❌ Processing failed: {error}")
                
        except Exception as e:
            print(f"   💥 Exception during processing: {str(e)}")
            img.status = 'failed'
            img.error_message = str(e)
            img.save()

def create_test_image():
    """Create a simple test image and process it"""
    print("\n🎨 Creating and processing a test image...")
    
    try:
        # Create a simple test image
        test_img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Add some shapes and colors
        cv2.rectangle(test_img, (50, 50), (150, 150), (255, 100, 100), -1)  # Blue rectangle
        cv2.circle(test_img, (200, 100), 50, (100, 255, 100), -1)  # Green circle
        cv2.ellipse(test_img, (150, 200), (80, 40), 0, 0, 360, (100, 100, 255), -1)  # Red ellipse
        
        # Save test image
        test_path = 'media/test_simple.jpg'
        os.makedirs('media', exist_ok=True)
        cv2.imwrite(test_path, test_img)
        
        # Test processing directly
        output_path = 'media/test_simple_output.jpg'
        success, processing_time, error = advanced_cartoon_processing(test_path, output_path, style='neural')
        
        if success:
            print(f"✅ Test processing completed in {processing_time:.2f}s")
            print(f"   Output saved: {output_path}")
        else:
            print(f"❌ Test processing failed: {error}")
            
    except Exception as e:
        print(f"💥 Test image creation/processing failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🚀 Starting comprehensive processing test...")
    print("=" * 60)
    
    # Test OpenCV first
    if test_opencv():
        # Process pending images
        process_all_pending()
        
        # Create and test simple image
        create_test_image()
    
    print("\n✨ Test completed!")
