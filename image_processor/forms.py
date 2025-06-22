from django import forms
from django.core.exceptions import ValidationError
from .models import ImageProcessing


class ImageUploadForm(forms.ModelForm):
    """Form for uploading images with validation"""
    
    CARTOON_STYLES = [
        ('neural', '🧠 Neural AI Style - Best quality with advanced processing (Recommended)'),
        ('classic', '🎨 Classic Cartoon - Bold edges and vibrant colors'),
        ('anime', '🌟 Anime Style - Smooth regions with vivid colors'),
        ('sketch', '✏️ Pencil Sketch - Artistic hand-drawn look'),
        ('watercolor', '🎭 Watercolor - Soft artistic painting effect'),
        ('oil_painting', '🖼️ Oil Painting - Rich textured artistic style'),
    ]
    
    cartoon_style = forms.ChoiceField(
        choices=CARTOON_STYLES,
        initial='neural',
        widget=forms.RadioSelect(attrs={
            'class': 'form-check-input'
        }),
        help_text="Choose your preferred cartoon animation style"
    )
    
    class Meta:
        model = ImageProcessing
        fields = ['original_image']
        widgets = {
            'original_image': forms.FileInput(attrs={
                'class': 'form-control-file',
                'accept': 'image/jpeg,image/jpg,image/png',
                'id': 'imageInput'
            })
        }
    
    def clean_original_image(self):
        """Validate uploaded image"""
        image = self.cleaned_data.get('original_image')
        
        if not image:
            raise ValidationError("Please select an image to upload.")
        
        # Check file size (5MB limit)
        if image.size > 5 * 1024 * 1024:
            raise ValidationError("Image file size must be less than 5MB.")
        
        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
        if image.content_type not in allowed_types:
            raise ValidationError("Only JPG and PNG files are allowed.")
        
        # Check file extension
        allowed_extensions = ['.jpg', '.jpeg', '.png']
        file_extension = image.name.lower().split('.')[-1]
        if f'.{file_extension}' not in allowed_extensions:
            raise ValidationError("Only JPG and PNG files are allowed.")
        
        return image
