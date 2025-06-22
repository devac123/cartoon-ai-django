"""
Celery tasks for background image processing
"""

try:
    from celery import shared_task
    CELERY_AVAILABLE = True
except ImportError:
    # Celery not installed, create a dummy decorator
    def shared_task(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    CELERY_AVAILABLE = False

from django.apps import apps
from .ai_utils import process_image_to_cartoon
import logging

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=3)
def process_image_task(self, image_processing_id):
    """
    Background task to process an image to cartoon style
    
    Args:
        image_processing_id (int): ID of the ImageProcessing object
    
    Returns:
        dict: Processing result with success status and message
    """
    try:
        # Get the ImageProcessing model
        ImageProcessing = apps.get_model('image_processor', 'ImageProcessing')
        
        # Get the image processing object
        image_processing = ImageProcessing.objects.get(id=image_processing_id)
        
        logger.info(f"Starting background processing for image {image_processing_id}")
        
        # Process the image
        success, error_msg = process_image_to_cartoon(image_processing)
        
        if success:
            logger.info(f"Successfully processed image {image_processing_id}")
            return {
                'success': True,
                'message': 'Image processed successfully',
                'image_id': image_processing_id
            }
        else:
            logger.error(f"Failed to process image {image_processing_id}: {error_msg}")
            return {
                'success': False,
                'message': error_msg,
                'image_id': image_processing_id
            }
            
    except ImageProcessing.DoesNotExist:
        error_msg = f"ImageProcessing object with id {image_processing_id} not found"
        logger.error(error_msg)
        return {
            'success': False,
            'message': error_msg,
            'image_id': image_processing_id
        }
        
    except Exception as exc:
        error_msg = f"Unexpected error processing image {image_processing_id}: {str(exc)}"
        logger.error(error_msg)
        
        # Retry the task if it's not the final attempt
        if self.request.retries < self.max_retries:
            logger.info(f"Retrying task for image {image_processing_id} (attempt {self.request.retries + 1})")
            raise self.retry(countdown=60, exc=exc)
        
        # Mark as failed after max retries
        try:
            ImageProcessing = apps.get_model('image_processor', 'ImageProcessing')
            image_processing = ImageProcessing.objects.get(id=image_processing_id)
            image_processing.status = 'failed'
            image_processing.error_message = error_msg
            image_processing.save()
        except:
            pass
        
        return {
            'success': False,
            'message': error_msg,
            'image_id': image_processing_id
        }


@shared_task
def cleanup_old_images():
    """
    Cleanup task to remove old processed images
    Run this task periodically to save disk space
    """
    from datetime import datetime, timedelta
    import os
    
    try:
        ImageProcessing = apps.get_model('image_processor', 'ImageProcessing')
        
        # Remove images older than 7 days
        cutoff_date = datetime.now() - timedelta(days=7)
        old_images = ImageProcessing.objects.filter(
            created_at__lt=cutoff_date
        )
        
        deleted_count = 0
        for image_processing in old_images:
            try:
                # Delete original image file
                if image_processing.original_image:
                    if os.path.exists(image_processing.original_image.path):
                        os.remove(image_processing.original_image.path)
                
                # Delete processed image file
                if image_processing.processed_image:
                    if os.path.exists(image_processing.processed_image.path):
                        os.remove(image_processing.processed_image.path)
                
                # Delete the database record
                image_processing.delete()
                deleted_count += 1
                
            except Exception as e:
                logger.error(f"Error deleting image {image_processing.id}: {str(e)}")
        
        logger.info(f"Cleanup completed. Deleted {deleted_count} old images.")
        return {
            'success': True,
            'deleted_count': deleted_count,
            'message': f'Successfully deleted {deleted_count} old images'
        }
        
    except Exception as e:
        error_msg = f"Cleanup task failed: {str(e)}"
        logger.error(error_msg)
        return {
            'success': False,
            'message': error_msg
        }
