#!/usr/bin/env python
"""
Debug script to check processing status
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cartoon_project.settings')
django.setup()

from image_processor.models import ImageProcessing
from image_processor.ai_utils import process_image_to_cartoon

def check_recent_images():
    """Check recent image processing attempts"""
    print("🔍 Checking recent image processing attempts...")
    print("=" * 60)
    
    recent_images = ImageProcessing.objects.all().order_by('-created_at')[:10]
    
    if not recent_images:
        print("❌ No images found in database")
        return
    
    for img in recent_images:
        print(f"ID: {img.id}")
        print(f"Status: {img.status}")
        print(f"Original: {img.original_image.name if img.original_image else 'None'}")
        print(f"Processed: {img.processed_image.name if img.processed_image else 'None'}")
        print(f"Error: {img.error_message or 'None'}")
        print(f"Created: {img.created_at}")
        print("-" * 40)

def test_processing():
    """Test the processing function with a simple image"""
    print("\n🧪 Testing processing function...")
    
    # Find the most recent pending/processing image
    pending_images = ImageProcessing.objects.filter(
        status__in=['pending', 'processing']
    ).order_by('-created_at')
    
    if pending_images:
        img = pending_images.first()
        print(f"📝 Found pending image ID: {img.id}")
        print(f"📁 Original file: {img.original_image.path}")
        
        # Check if file exists
        if os.path.exists(img.original_image.path):
            print("✅ Original file exists")
            
            # Try processing
            print("🔄 Attempting to process...")
            try:
                success, error = process_image_to_cartoon(img)
                print(f"✅ Processing result: Success={success}")
                if error:
                    print(f"❌ Error: {error}")
                else:
                    print("🎉 Processing completed successfully!")
            except Exception as e:
                print(f"💥 Processing failed with exception: {str(e)}")
                import traceback
                traceback.print_exc()
        else:
            print("❌ Original file not found!")
    else:
        print("📭 No pending images to process")

if __name__ == '__main__':
    check_recent_images()
    test_processing()
