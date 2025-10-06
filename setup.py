#!/usr/bin/env python3
"""
Setup script for Advanced POS Tagger
====================================

This script sets up the environment and downloads required models.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """Install required packages"""
    return run_command("pip install -r requirements.txt", "Installing Python packages")

def download_spacy_model():
    """Download SpaCy English model"""
    return run_command("python -m spacy download en_core_web_sm", "Downloading SpaCy English model")

def download_nltk_data():
    """Download NLTK data"""
    nltk_commands = [
        "python -c \"import nltk; nltk.download('punkt')\"",
        "python -c \"import nltk; nltk.download('averaged_perceptron_tagger')\"",
        "python -c \"import nltk; nltk.download('tagsets')\""
    ]
    
    for cmd in nltk_commands:
        if not run_command(cmd, "Downloading NLTK data"):
            return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ["logs", "temp", "models"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    return True

def test_installation():
    """Test if installation was successful"""
    print("🧪 Testing installation...")
    try:
        from pos_tagger import AdvancedPOSTagger
        tagger = AdvancedPOSTagger()
        print("✅ POS tagger imported successfully")
        
        # Test basic functionality
        test_text = "Hello world"
        analysis = tagger.tag_text(test_text, "spacy_en")
        print(f"✅ Basic functionality test passed: {len(analysis.tokens)} tokens processed")
        
        return True
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Advanced POS Tagger")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Download SpaCy model
    if not download_spacy_model():
        print("⚠️  SpaCy model download failed - some features may not work")
    
    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  NLTK data download failed - some features may not work")
    
    # Create directories
    create_directories()
    
    # Test installation
    if test_installation():
        print("\n🎉 Setup completed successfully!")
        print("\n📖 Next steps:")
        print("1. Run the web interface: streamlit run app.py")
        print("2. Run the command line demo: python pos_tagger.py")
        print("3. Run tests: python -m pytest test_pos_tagger.py -v")
    else:
        print("\n❌ Setup completed with errors")
        print("Please check the error messages above and try again")
        sys.exit(1)

if __name__ == "__main__":
    main()
