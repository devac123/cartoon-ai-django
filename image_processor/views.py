from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.urls import reverse
from django.conf import settings
from django.views.generic import TemplateView
from .models import ImageProcessing
from .forms import ImageUploadForm
from .ai_utils import process_image_to_cartoon, process_image_with_style, advanced_cartoon_processing
import os
import mimetypes


class HomeView(TemplateView):
    """Homepage with service description and upload form"""
    template_name = 'image_processor/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['form'] = ImageUploadForm()
        context['upi_id'] = settings.UPI_ID
        context['telegram_handle'] = settings.TELEGRAM_HANDLE
        return context


def upload_image(request):
    """Handle image upload and processing"""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Save the image processing object
                image_processing = form.save()
                
                # Get the selected cartoon style
                cartoon_style = form.cleaned_data.get('cartoon_style', 'neural')
                
                # Process the image with the selected style
                success, error_msg = process_image_with_style(image_processing, cartoon_style)
                
                if success:
                    messages.success(request, f'Image processed successfully with {cartoon_style} style!')
                    return redirect('image_processor:result', pk=image_processing.pk)
                else:
                    messages.error(request, f'Processing failed: {error_msg}')
                    return redirect('image_processor:home')
                    
            except Exception as e:
                messages.error(request, f'Upload failed: {str(e)}')
                return redirect('image_processor:home')
        else:
            # Form has validation errors
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, error)
            return redirect('image_processor:home')
    else:
        return redirect('image_processor:home')


def result_view(request, pk):
    """Display processing result"""
    image_processing = get_object_or_404(ImageProcessing, pk=pk)
    
    context = {
        'image_processing': image_processing,
        'upi_id': settings.UPI_ID,
        'telegram_handle': settings.TELEGRAM_HANDLE,
    }
    
    return render(request, 'image_processor/result.html', context)


def download_image(request, pk):
    """Download processed image"""
    image_processing = get_object_or_404(ImageProcessing, pk=pk)
    
    if not image_processing.processed_image:
        messages.error(request, 'No processed image available for download.')
        return redirect('image_processor:result', pk=pk)
    
    try:
        # Get the file path
        file_path = image_processing.processed_image.path
        
        if os.path.exists(file_path):
            # Determine content type
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            # Read file content
            with open(file_path, 'rb') as f:
                response = HttpResponse(f.read(), content_type=content_type)
                response['Content-Disposition'] = f'attachment; filename="{image_processing.processed_filename}"'
                return response
        else:
            messages.error(request, 'File not found.')
            return redirect('image_processor:result', pk=pk)
            
    except Exception as e:
        messages.error(request, f'Download failed: {str(e)}')
        return redirect('image_processor:result', pk=pk)


def check_status(request, pk):
    """AJAX endpoint to check processing status"""
    try:
        image_processing = get_object_or_404(ImageProcessing, pk=pk)
        
        data = {
            'status': image_processing.status,
            'processing_time': image_processing.processing_time,
            'error_message': image_processing.error_message,
        }
        
        if image_processing.status == 'completed':
            data['result_url'] = reverse('image_processor:result', kwargs={'pk': pk})
        
        return JsonResponse(data)
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def gallery_view(request):
    """Display recent processed images (optional feature)"""
    recent_images = ImageProcessing.objects.filter(
        status='completed'
    ).order_by('-created_at')[:12]
    
    context = {
        'recent_images': recent_images,
    }
    
    return render(request, 'image_processor/gallery.html', context)
