#!/usr/bin/env bash
# Build script for Render.com deployment

set -o errexit  # exit on error

echo "🚀 Starting build process for Cartoon AI..."

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "🗃️ Running database migrations..."
python manage.py migrate --noinput

echo "📁 Collecting static files..."
python manage.py collectstatic --noinput --clear

echo "✅ Build completed successfully!"
echo "🎨 Cartoon AI is ready for deployment!"
