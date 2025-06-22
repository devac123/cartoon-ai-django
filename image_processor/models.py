from django.db import models
from django.urls import reverse
import os


def upload_to_original(instance, filename):
    """Generate upload path for original images"""
    return f'uploads/original/{filename}'


def upload_to_processed(instance, filename):
    """Generate upload path for processed images"""
    name, ext = os.path.splitext(filename)
    return f'uploads/processed/{name}_cartoon{ext}'


class ImageProcessing(models.Model):
    """Model to store image processing requests and results"""
    
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    
    original_image = models.ImageField(
        upload_to=upload_to_original,
        help_text="Original image uploaded by user"
    )
    processed_image = models.ImageField(
        upload_to=upload_to_processed,
        blank=True,
        null=True,
        help_text="AI-processed cartoon image"
    )
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='pending'
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processing_time = models.FloatField(
        null=True,
        blank=True,
        help_text="Time taken to process in seconds"
    )
    error_message = models.TextField(
        blank=True,
        null=True,
        help_text="Error message if processing failed"
    )
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Image Processing'
        verbose_name_plural = 'Image Processings'
    
    def __str__(self):
        return f"Image {self.id} - {self.status}"
    
    def get_absolute_url(self):
        return reverse('image_processor:result', kwargs={'pk': self.pk})
    
    @property
    def original_filename(self):
        """Get original filename without path"""
        if self.original_image:
            return os.path.basename(self.original_image.name)
        return None
    
    @property
    def processed_filename(self):
        """Get processed filename without path"""
        if self.processed_image:
            return os.path.basename(self.processed_image.name)
        return None
