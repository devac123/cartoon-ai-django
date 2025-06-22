#!/usr/bin/env python
"""
Simulate an upload to test the entire pipeline
"""

import os
import sys
import django
from django.core.files.uploadedfile import SimpleUploadedFile

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cartoon_project.settings')
django.setup()

from image_processor.models import ImageProcessing
from image_processor.forms import ImageUploadForm
from image_processor.ai_utils import process_image_to_cartoon
import cv2
import numpy as np

def create_test_upload():
    """Create a test image and simulate the upload process"""
    print("🧪 Testing complete upload pipeline...")
    print("=" * 60)
    
    try:
        # Create a test image
        print("1️⃣ Creating test image...")
        test_img = np.zeros((400, 400, 3), dtype=np.uint8)
        
        # Add colorful content
        cv2.rectangle(test_img, (50, 50), (200, 200), (255, 100, 100), -1)
        cv2.circle(test_img, (300, 150), 80, (100, 255, 100), -1)
        cv2.ellipse(test_img, (200, 300), (120, 60), 0, 0, 360, (100, 100, 255), -1)
        
        # Add some text-like shapes
        cv2.putText(test_img, 'TEST', (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save as bytes
        _, img_bytes = cv2.imencode('.jpg', test_img)
        img_data = img_bytes.tobytes()
        
        print("✅ Test image created")
        
        # Simulate form upload
        print("\n2️⃣ Simulating form upload...")
        uploaded_file = SimpleUploadedFile(
            "test_upload.jpg",
            img_data,
            content_type="image/jpeg"
        )
        
        form_data = {'cartoon_style': 'neural'}
        file_data = {'original_image': uploaded_file}
        form = ImageUploadForm(form_data, file_data)
        
        if form.is_valid():
            print("✅ Form validation passed")
            
            # Save the form (creates ImageProcessing object)
            print("\n3️⃣ Saving to database...")
            image_processing = form.save()
            print(f"✅ Created ImageProcessing object with ID: {image_processing.id}")
            print(f"   Status: {image_processing.status}")
            print(f"   File: {image_processing.original_image.name}")
            
            # Process the image
            print("\n4️⃣ Processing image...")
            success, error = process_image_to_cartoon(image_processing)
            
            if success:
                print("✅ Processing completed successfully!")
                print(f"   Status: {image_processing.status}")
                print(f"   Processing time: {image_processing.processing_time:.2f}s")
                if image_processing.processed_image:
                    print(f"   Output file: {image_processing.processed_image.name}")
                
                return image_processing.id
            else:
                print(f"❌ Processing failed: {error}")
                return None
                
        else:
            print("❌ Form validation failed:")
            for field, errors in form.errors.items():
                for error in errors:
                    print(f"   {field}: {error}")
            return None
            
    except Exception as e:
        print(f"💥 Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def verify_result(processing_id):
    """Verify the processing result"""
    if not processing_id:
        return
        
    print(f"\n🔍 Verifying result for ID {processing_id}...")
    
    try:
        img = ImageProcessing.objects.get(id=processing_id)
        print(f"Status: {img.status}")
        print(f"Original file exists: {os.path.exists(img.original_image.path) if img.original_image else False}")
        print(f"Processed file exists: {os.path.exists(img.processed_image.path) if img.processed_image else False}")
        
        if img.processed_image and os.path.exists(img.processed_image.path):
            print("✅ Complete success! Both files exist and processing completed.")
        else:
            print("⚠️ Processing may have issues - missing processed file")
            
    except Exception as e:
        print(f"❌ Verification failed: {str(e)}")

if __name__ == '__main__':
    print("🚀 Starting complete upload simulation...")
    processing_id = create_test_upload()
    verify_result(processing_id)
    print("\n✨ Upload simulation completed!")
