#!/usr/bin/env python3
"""
Setup script for Employee Review Analytics Dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False

def check_data_file():
    """Check if the merged_reviews.json file exists"""
    if os.path.exists('merged_reviews.json'):
        print("✅ Data file found: merged_reviews.json")
        return True
    else:
        print("❌ Data file not found: merged_reviews.json")
        print("Please ensure the merged_reviews.json file is in the same directory as this script.")
        return False

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("Launching the dashboard...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def main():
    """Main setup function"""
    print("🚀 Setting up Employee Review Analytics Dashboard...")
    print("=" * 50)
    
    # Check if data file exists
    if not check_data_file():
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Download NLTK data
    if not download_nltk_data():
        return
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("🌟 Starting the dashboard...")
    print("=" * 50)
    
    # Run the dashboard
    run_dashboard()

if __name__ == "__main__":
    main()
