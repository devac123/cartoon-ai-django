from django.urls import path
from . import views

app_name = 'image_processor'

urlpatterns = [
    # Main pages
    path('', views.HomeView.as_view(), name='home'),
    path('upload/', views.upload_image, name='upload'),
    path('result/<int:pk>/', views.result_view, name='result'),
    path('download/<int:pk>/', views.download_image, name='download'),
    path('gallery/', views.gallery_view, name='gallery'),
    
    # AJAX endpoints
    path('status/<int:pk>/', views.check_status, name='check_status'),
]
