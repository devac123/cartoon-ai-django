from django.test import TestCase, Client
from django.urls import reverse
from django.core.files.uploadedfile import SimpleUploadedFile
from django.contrib.auth.models import User
from PIL import Image
import io
import os
from .models import ImageProcessing
from .forms import ImageUploadForm
from .ai_utils import mock_cartoon_processing, process_image_to_cartoon


class ImageProcessingModelTest(TestCase):
    """Test cases for ImageProcessing model"""
    
    def setUp(self):
        """Set up test data"""
        # Create a test image
        self.test_image = self.create_test_image()
        
    def create_test_image(self, format='JPEG'):
        """Create a test image file"""
        file = io.BytesIO()
        image = Image.new('RGB', (100, 100), color='red')
        image.save(file, format=format)
        file.seek(0)
        return SimpleUploadedFile(
            f'test_image.{format.lower()}',
            file.getvalue(),
            content_type=f'image/{format.lower()}'
        )
    
    def test_image_processing_creation(self):
        """Test creating an ImageProcessing instance"""
        image_processing = ImageProcessing.objects.create(
            original_image=self.test_image
        )
        
        self.assertEqual(image_processing.status, 'pending')
        self.assertIsNotNone(image_processing.created_at)
        self.assertIsNotNone(image_processing.updated_at)
        self.assertFalse(image_processing.processed_image)
    
    def test_image_processing_str_method(self):
        """Test string representation of ImageProcessing"""
        image_processing = ImageProcessing.objects.create(
            original_image=self.test_image
        )
        
        expected_str = f"Image {image_processing.id} - pending"
        self.assertEqual(str(image_processing), expected_str)
    
    def test_original_filename_property(self):
        """Test original_filename property"""
        image_processing = ImageProcessing.objects.create(
            original_image=self.test_image
        )
        
        self.assertIn('test_image', image_processing.original_filename)


class ImageUploadFormTest(TestCase):
    """Test cases for ImageUploadForm"""
    
    def create_test_image(self, size=(100, 100), format='JPEG'):
        """Create a test image file"""
        file = io.BytesIO()
        image = Image.new('RGB', size, color='blue')
        image.save(file, format=format)
        file.seek(0)
        return SimpleUploadedFile(
            f'test.{format.lower()}',
            file.getvalue(),
            content_type=f'image/{format.lower()}'
        )
    
    def test_valid_image_upload(self):
        """Test uploading a valid image"""
        image_file = self.create_test_image()
        form_data = {}
        file_data = {'original_image': image_file}
        
        form = ImageUploadForm(form_data, file_data)
        self.assertTrue(form.is_valid())
    
    def test_invalid_file_type(self):
        """Test uploading an invalid file type"""
        # Create a text file instead of image
        invalid_file = SimpleUploadedFile(
            'test.txt',
            b'This is not an image',
            content_type='text/plain'
        )
        
        form_data = {}
        file_data = {'original_image': invalid_file}
        
        form = ImageUploadForm(form_data, file_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Only JPG and PNG files are allowed', str(form.errors))
    
    def test_file_size_validation(self):
        """Test file size validation (mock large file)"""
        # Create a large image (simulate 6MB by setting size attribute)
        large_image = self.create_test_image()
        large_image.size = 6 * 1024 * 1024  # 6MB
        
        form_data = {}
        file_data = {'original_image': large_image}
        
        form = ImageUploadForm(form_data, file_data)
        self.assertFalse(form.is_valid())
        self.assertIn('Image file size must be less than 5MB', str(form.errors))
    
    def test_png_file_upload(self):
        """Test uploading a PNG file"""
        png_image = self.create_test_image(format='PNG')
        form_data = {}
        file_data = {'original_image': png_image}
        
        form = ImageUploadForm(form_data, file_data)
        self.assertTrue(form.is_valid())


class ViewsTest(TestCase):
    """Test cases for views"""
    
    def setUp(self):
        """Set up test client and test data"""
        self.client = Client()
        self.test_image = self.create_test_image()
    
    def create_test_image(self):
        """Create a test image file"""
        file = io.BytesIO()
        image = Image.new('RGB', (100, 100), color='green')
        image.save(file, 'JPEG')
        file.seek(0)
        return SimpleUploadedFile(
            'test_view.jpg',
            file.getvalue(),
            content_type='image/jpeg'
        )
    
    def test_home_view(self):
        """Test home page loads correctly"""
        response = self.client.get(reverse('image_processor:home'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Transform Your Photos')
        self.assertContains(response, 'Upload Your Photo')
    
    def test_gallery_view(self):
        """Test gallery page loads correctly"""
        response = self.client.get(reverse('image_processor:gallery'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Gallery')
    
    def test_image_upload_post(self):
        """Test image upload via POST request"""
        response = self.client.post(
            reverse('image_processor:upload'),
            {'original_image': self.test_image},
            follow=True
        )
        
        # Should redirect to result page after successful upload
        self.assertEqual(response.status_code, 200)
        
        # Check that an ImageProcessing object was created
        self.assertEqual(ImageProcessing.objects.count(), 1)
        
        # Check the object status
        image_processing = ImageProcessing.objects.first()
        self.assertIn(image_processing.status, ['completed', 'processing', 'pending'])
    
    def test_upload_invalid_file(self):
        """Test uploading an invalid file"""
        invalid_file = SimpleUploadedFile(
            'invalid.txt',
            b'This is not an image',
            content_type='text/plain'
        )
        
        response = self.client.post(
            reverse('image_processor:upload'),
            {'original_image': invalid_file},
            follow=True
        )
        
        # Should redirect back to home with error message
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Only JPG and PNG files are allowed')
    
    def test_result_view(self):
        """Test result view with a processed image"""
        # Create an ImageProcessing object
        image_processing = ImageProcessing.objects.create(
            original_image=self.test_image,
            status='completed'
        )
        
        response = self.client.get(
            reverse('image_processor:result', kwargs={'pk': image_processing.pk})
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Your Cartoon is Ready')
    
    def test_result_view_404(self):
        """Test result view with non-existent ID"""
        response = self.client.get(
            reverse('image_processor:result', kwargs={'pk': 999})
        )
        
        self.assertEqual(response.status_code, 404)
    
    def test_download_view(self):
        """Test download functionality"""
        # Create an ImageProcessing object with processed image
        image_processing = ImageProcessing.objects.create(
            original_image=self.test_image,
            status='completed'
        )
        
        response = self.client.get(
            reverse('image_processor:download', kwargs={'pk': image_processing.pk})
        )
        
        # Since no processed image exists, should redirect with error
        self.assertEqual(response.status_code, 302)
    
    def test_status_check_ajax(self):
        """Test AJAX status check endpoint"""
        image_processing = ImageProcessing.objects.create(
            original_image=self.test_image,
            status='processing'
        )
        
        response = self.client.get(
            reverse('image_processor:check_status', kwargs={'pk': image_processing.pk}),
            HTTP_X_REQUESTED_WITH='XMLHttpRequest'
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/json')
        
        # Parse JSON response
        import json
        data = json.loads(response.content)
        self.assertEqual(data['status'], 'processing')


class AIUtilsTest(TestCase):
    """Test cases for AI processing utilities"""
    
    def create_test_image_file(self):
        """Create a temporary test image file"""
        import tempfile
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        
        # Create and save test image
        image = Image.new('RGB', (200, 200), color='yellow')
        image.save(temp_file.name, 'JPEG')
        
        return temp_file.name
    
    def test_mock_cartoon_processing(self):
        """Test mock cartoon processing function"""
        input_path = self.create_test_image_file()
        output_path = input_path.replace('.jpg', '_cartoon.jpg')
        
        try:
            success, processing_time, error_msg = mock_cartoon_processing(
                input_path, output_path
            )
            
            self.assertTrue(success)
            self.assertIsNone(error_msg)
            self.assertGreater(processing_time, 0)
            self.assertTrue(os.path.exists(output_path))
            
        finally:
            # Clean up temporary files
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(output_path):
                os.remove(output_path)
    
    def test_mock_processing_with_invalid_input(self):
        """Test mock processing with invalid input"""
        invalid_path = '/nonexistent/path/image.jpg'
        output_path = '/tmp/output.jpg'
        
        success, processing_time, error_msg = mock_cartoon_processing(
            invalid_path, output_path
        )
        
        self.assertFalse(success)
        self.assertEqual(processing_time, 0)
        self.assertIsNotNone(error_msg)


class IntegrationTest(TestCase):
    """Integration tests for the complete workflow"""
    
    def create_test_image(self):
        """Create a test image file"""
        file = io.BytesIO()
        image = Image.new('RGB', (150, 150), color='purple')
        image.save(file, 'JPEG')
        file.seek(0)
        return SimpleUploadedFile(
            'integration_test.jpg',
            file.getvalue(),
            content_type='image/jpeg'
        )
    
    def test_complete_workflow(self):
        """Test the complete image processing workflow"""
        # 1. Upload image
        test_image = self.create_test_image()
        response = self.client.post(
            reverse('image_processor:upload'),
            {'original_image': test_image}
        )
        
        # Should redirect to result page
        self.assertEqual(response.status_code, 302)
        
        # 2. Check that ImageProcessing object was created
        self.assertEqual(ImageProcessing.objects.count(), 1)
        image_processing = ImageProcessing.objects.first()
        
        # 3. Check result page
        response = self.client.get(
            reverse('image_processor:result', kwargs={'pk': image_processing.pk})
        )
        self.assertEqual(response.status_code, 200)
        
        # 4. Test status check
        response = self.client.get(
            reverse('image_processor:check_status', kwargs={'pk': image_processing.pk})
        )
        self.assertEqual(response.status_code, 200)
        
        # 5. If processing completed, test download
        if image_processing.status == 'completed' and image_processing.processed_image:
            response = self.client.get(
                reverse('image_processor:download', kwargs={'pk': image_processing.pk})
            )
            # Should either download file or redirect with error
            self.assertIn(response.status_code, [200, 302])
