#!/usr/bin/env python
"""
Manual processing trigger for debugging stuck images
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cartoon_project.settings')
django.setup()

from image_processor.models import ImageProcessing
from image_processor.ai_utils import process_image_to_cartoon

def manual_process_all():
    """Manually process all non-completed images"""
    print("🔧 Manual Processing Tool")
    print("=" * 50)
    
    # Find all non-completed images
    images_to_process = ImageProcessing.objects.exclude(status='completed')
    
    if not images_to_process:
        print("✅ No images need processing")
        
        # Show all recent images
        all_images = ImageProcessing.objects.all().order_by('-created_at')[:5]
        if all_images:
            print("\n📋 Recent images:")
            for img in all_images:
                status_emoji = "✅" if img.status == "completed" else "⏳" if img.status == "processing" else "❌"
                print(f"  {status_emoji} ID: {img.id} | Status: {img.status} | File: {os.path.basename(img.original_image.name) if img.original_image else 'None'}")
        return
    
    print(f"🔄 Found {images_to_process.count()} images to process")
    
    for img in images_to_process:
        print(f"\n📝 Processing ID: {img.id}")
        print(f"   File: {img.original_image.name if img.original_image else 'None'}")
        print(f"   Current Status: {img.status}")
        
        if not img.original_image:
            print("   ❌ No original image file")
            img.status = 'failed'
            img.error_message = 'No original image file'
            img.save()
            continue
            
        if not os.path.exists(img.original_image.path):
            print(f"   ❌ File not found: {img.original_image.path}")
            img.status = 'failed'
            img.error_message = 'Original file not found'
            img.save()
            continue
        
        # Force processing
        print("   🚀 Starting processing...")
        img.status = 'processing'
        img.save()
        
        try:
            success, error = process_image_to_cartoon(img)
            
            if success:
                print("   ✅ SUCCESS! Processing completed")
            else:
                print(f"   ❌ FAILED: {error}")
                
        except Exception as e:
            print(f"   💥 EXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    manual_process_all()
    print("\n🎉 Manual processing completed!")
