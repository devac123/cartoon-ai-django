#!/usr/bin/env python
"""
Quick startup script for the Cartoon AI Django application
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def main():
    """Main startup function"""
    print("🎨 Cartoon AI Django Application Startup Script")
    print("=" * 50)
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Virtual environment not detected!")
        print("Please activate your virtual environment first:")
        print("   Windows: venv\\Scripts\\activate")
        print("   Linux/Mac: source venv/bin/activate")
        response = input("\nDo you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            return
    
    # Check if .env file exists
    if not Path('.env').exists():
        print("📄 Creating .env file from template...")
        if Path('.env.example').exists():
            try:
                with open('.env.example', 'r') as src, open('.env', 'w') as dst:
                    dst.write(src.read())
                print("✅ .env file created successfully")
                print("⚠️  Please edit .env file with your actual configuration!")
            except Exception as e:
                print(f"❌ Failed to create .env file: {e}")
        else:
            print("❌ .env.example not found!")
    
    # Run migrations
    if not run_command("python manage.py migrate", "Running database migrations"):
        return
    
    # Collect static files
    if not run_command("python manage.py collectstatic --noinput", "Collecting static files"):
        print("⚠️  Static files collection failed, but continuing...")
    
    # Check if superuser exists (optional)
    print("\n🔑 Checking admin user...")
    try:
        result = subprocess.run(
            "python manage.py shell -c \"from django.contrib.auth.models import User; print('exists' if User.objects.filter(is_superuser=True).exists() else 'none')\"",
            shell=True, capture_output=True, text=True
        )
        if 'none' in result.stdout:
            print("ℹ️  No superuser found. You can create one later with:")
            print("   python manage.py createsuperuser")
        else:
            print("✅ Superuser already exists")
    except:
        print("⚠️  Could not check superuser status")
    
    print("\n🚀 Starting development server...")
    print("📱 Application will be available at: http://127.0.0.1:8000/")
    print("🛑 Press CTRL+C to stop the server")
    print("=" * 50)
    
    try:
        subprocess.run("python manage.py runserver", shell=True, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped. Thank you for using Cartoon AI!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server failed to start: {e}")

if __name__ == "__main__":
    main()
