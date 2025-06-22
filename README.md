# Cartoon AI - Photo to Cartoon Converter

A Django web application that transforms photos into cartoon-style images using AI technology. Users can upload JPG/PNG images and get instant cartoon conversions with optional HD upgrades via UPI payment.

## 🚀 Features

### Core Features
- **Image Upload**: Support for JPG and PNG files up to 5MB
- **AI Processing**: Mock cartoon-style transformation (ready for real AI integration)
- **Responsive UI**: Professional design built with Bootstrap 5
- **File Validation**: Comprehensive file type and size validation
- **Download System**: Easy download of processed images
- **Gallery**: Showcase of recent transformations

### Business Features
- **UPI Payment Integration**: ₹49 payment for HD version via UPI
- **Telegram Contact**: Direct Telegram handle for customer support
- **Mobile Responsive**: Works perfectly on all devices
- **SEO Optimized**: Meta tags and proper structure

### Technical Features
- **Django Framework**: Python web framework
- **SQLite Database**: Default database (production-ready)
- **Media Management**: Proper static/media file handling
- **WhiteNoise**: Static file serving for deployment
- **Celery Support**: Background processing capability
- **Comprehensive Tests**: Full test suite included
- **Error Handling**: Robust error handling and user feedback

## 📋 Requirements

- Python 3.11+
- Django 5.2+
- Pillow (for image processing)
- Bootstrap 5 (CDN)
- Optional: Redis (for Celery background tasks)

## 🛠️ Local Development Setup

### 1. Clone and Setup Virtual Environment

```bash
# Create project directory
mkdir cartoon-app
cd cartoon-app

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configurations
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,.render.com
UPI_ID=yourupi@bank
TELEGRAM_HANDLE=@yourTelegramHandle
```

### 4. Database Setup

```bash
# Run migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser (optional)
python manage.py createsuperuser
```

### 5. Collect Static Files

```bash
python manage.py collectstatic
```

### 6. Run Development Server

```bash
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to see the application.

## 🧪 Running Tests

```bash
# Run all tests
python manage.py test

# Run specific test file
python manage.py test image_processor.tests

# Run with coverage (if installed)
coverage run --source='.' manage.py test
coverage report
```

## 📁 Project Structure

```
cartoon-app/
├── cartoon_project/           # Main Django project
│   ├── __init__.py
│   ├── settings.py           # Django settings
│   ├── urls.py              # Main URL configuration
│   ├── wsgi.py              # WSGI configuration
│   └── celery.py            # Celery configuration
├── image_processor/          # Main app
│   ├── migrations/          # Database migrations
│   ├── templates/           # HTML templates
│   │   └── image_processor/
│   │       ├── base.html    # Base template
│   │       ├── home.html    # Homepage
│   │       ├── result.html  # Result page
│   │       └── gallery.html # Gallery page
│   ├── __init__.py
│   ├── admin.py            # Django admin config
│   ├── apps.py
│   ├── forms.py            # Django forms
│   ├── models.py           # Database models
│   ├── views.py            # View functions
│   ├── urls.py             # App URL patterns
│   ├── ai_utils.py         # AI processing utilities
│   ├── tasks.py            # Celery tasks
│   └── tests.py            # Test cases
├── static/                  # Static files
│   ├── css/
│   ├── js/
│   └── img/
├── media/                   # User uploaded files
├── venv/                    # Virtual environment
├── .env                     # Environment variables
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
├── Procfile               # Render.com deployment
├── runtime.txt            # Python version
├── manage.py              # Django management
└── README.md              # This file
```

## 🎨 UI Components

### Homepage
- Hero section with service description
- Drag & drop image upload
- File validation and preview
- Features showcase
- UPI payment CTA

### Result Page
- Before/after image comparison
- Download functionality
- Processing status updates
- HD upgrade promotion
- Social sharing options

### Gallery
- Recent transformations showcase
- Before/after previews
- Processing time display
- Empty state handling

## 🔧 Configuration Options

### Django Settings (`settings.py`)
- Media file configuration
- Static file handling with WhiteNoise
- File upload limits (5MB)
- Celery configuration
- Custom UPI/Telegram settings

### Environment Variables (`.env`)
```env
SECRET_KEY=your-django-secret-key
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,.render.com
REDIS_URL=redis://localhost:6379/0
UPI_ID=yourupi@bank
TELEGRAM_HANDLE=@yourTelegramHandle
```

## 🤖 AI Integration

### Current Implementation
The app includes a mock AI processing function that applies basic image filters to simulate cartoon effects:
- Color enhancement
- Blur and sharpen effects
- Contrast adjustment

### Real AI Integration
To integrate real AI models like Toonify or AnimeGAN:

1. **Toonify API Integration**:
```python
# In ai_utils.py
def integrate_toonify_api(image_path, api_key):
    # Implement Toonify API calls
    # Documentation: https://toonify.photos/
    pass
```

2. **AnimeGAN Model**:
```python
# In ai_utils.py
def integrate_animegan_model(image_path, model_path):
    # Implement AnimeGAN model processing
    # GitHub: https://github.com/TachibanaYoshino/AnimeGAN
    pass
```

3. **Other AI Services**:
- DeepAI Toonify API
- Runway ML
- Custom trained models

## 📱 Mobile Optimization

- Responsive Bootstrap 5 design
- Touch-friendly upload area
- Optimized images for mobile
- Fast loading times
- Progressive Web App ready

## 🚀 Deployment

### Render.com Deployment

1. **Push to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin your-repo-url
git push -u origin main
```

2. **Create Render Service**:
   - Connect GitHub repository
   - Choose "Web Service"
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `gunicorn cartoon_project.wsgi:application`

3. **Environment Variables**:
   - Add all variables from `.env.example`
   - Set `DEBUG=False`
   - Set `ALLOWED_HOSTS` to include your Render domain

4. **Database**:
   - SQLite works for small applications
   - Consider PostgreSQL for production

### Other Deployment Options

#### Heroku
```bash
# Install Heroku CLI and login
heroku create your-app-name
heroku config:set SECRET_KEY=your-secret-key
heroku config:set DEBUG=False
git push heroku main
```

#### DigitalOcean App Platform
- Connect GitHub repository
- Configure environment variables
- Deploy with automatic builds

#### Traditional VPS
```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip nginx supervisor

# Setup application
git clone your-repo
cd cartoon-app
pip install -r requirements.txt

# Configure Nginx and Supervisor
# ... (detailed configuration files needed)
```

## 🔄 Background Processing (Optional)

### Celery Setup

1. **Install Redis**:
```bash
# Ubuntu/Debian
sudo apt install redis-server

# macOS
brew install redis

# Windows
# Download from https://redis.io/download
```

2. **Start Celery Worker**:
```bash
celery -A cartoon_project worker --loglevel=info
```

3. **Use Background Processing**:
```python
# In views.py
from .tasks import process_image_task

# Instead of synchronous processing
task = process_image_task.delay(image_processing.id)
```

## 📊 Monitoring and Analytics

### Recommended Tools
- **Sentry**: Error tracking
- **Google Analytics**: User behavior
- **Uptime Robot**: Service monitoring
- **LogRocket**: Session replay

### Custom Metrics
- Upload success rate
- Processing time statistics
- User conversion (free to paid)
- Popular image types

## 🔒 Security Considerations

### File Upload Security
- File type validation
- File size limits
- Virus scanning (production)
- Sandboxed processing

### Django Security
- CSRF protection
- SQL injection prevention
- XSS protection
- Secure headers

### Production Security
```python
# In production settings
SECURE_SSL_REDIRECT = True
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
X_FRAME_OPTIONS = 'DENY'
```

## 🛡️ Error Handling

### User-Friendly Messages
- Clear upload error messages
- Processing failure explanations
- Network error recovery
- Graceful degradation

### Logging
```python
import logging
logger = logging.getLogger(__name__)

# Log important events
logger.info("Image processing started")
logger.error("Processing failed", exc_info=True)
```

## 📈 Performance Optimization

### Image Optimization
- Automatic image resizing
- Format optimization
- Progressive JPEG loading
- WebP format support

### Caching
```python
# Add caching for static content
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}
```

### Database Optimization
- Query optimization
- Index creation
- Connection pooling
- Regular cleanup tasks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guide
- Write comprehensive tests
- Update documentation
- Use meaningful commit messages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Documentation
- Django: https://docs.djangoproject.com/
- Bootstrap: https://getbootstrap.com/docs/
- Celery: https://docs.celeryproject.org/

### Community
- Stack Overflow: Tag with `django` and `image-processing`
- Django Forum: https://forum.djangoproject.com/
- Reddit: r/django

### Contact
- Telegram: Your configured handle
- Email: your-email@domain.com
- GitHub Issues: Use the repository issue tracker

---

**Happy Coding! 🎨✨**
