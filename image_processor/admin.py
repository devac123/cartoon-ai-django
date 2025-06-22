from django.contrib import admin
from .models import ImageProcessing


@admin.register(ImageProcessing)
class ImageProcessingAdmin(admin.ModelAdmin):
    """Admin configuration for ImageProcessing model"""
    
    list_display = [
        'id',
        'status',
        'original_filename',
        'processing_time',
        'created_at',
        'updated_at'
    ]
    
    list_filter = [
        'status',
        'created_at',
        'updated_at'
    ]
    
    search_fields = [
        'original_image',
        'processed_image'
    ]
    
    readonly_fields = [
        'created_at',
        'updated_at',
        'processing_time'
    ]
    
    fieldsets = [
        ('Images', {
            'fields': ('original_image', 'processed_image')
        }),
        ('Processing Info', {
            'fields': ('status', 'processing_time', 'error_message')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    ]
    
    def original_filename(self, obj):
        """Display original filename in admin"""
        return obj.original_filename or 'N/A'
    original_filename.short_description = 'Original File'
    
    def get_queryset(self, request):
        """Optimize queryset for admin"""
        return super().get_queryset(request).select_related()
