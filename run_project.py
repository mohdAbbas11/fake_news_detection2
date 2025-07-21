import os
import sys
import subprocess
import argparse
import time

# Set paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, 'data')
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
APP_DIR = os.path.join(CURRENT_DIR, 'app')

# Function to check if Python package is installed
def is_package_installed(package_name):
    """
    Check if a Python package is installed
    
    Args:
        package_name (str): Name of the package to check
        
    Returns:
        bool: True if installed, False otherwise
    """
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# Function to install requirements
def install_requirements():
    """
    Install required packages from requirements.txt
    """
    print("\n===== Installing Requirements =====")
    requirements_path = os.path.join(CURRENT_DIR, 'requirements.txt')
    
    if os.path.exists(requirements_path):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', requirements_path])
            print("Requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error installing requirements: {e}")
            return False
    else:
        print("requirements.txt not found. Skipping installation.")
    
    return True

# Function to download NLTK data
def download_nltk_data():
    """
    Download required NLTK data
    """
    print("\n===== Downloading NLTK Data =====")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        print("NLTK data downloaded successfully!")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False
    
    return True

# Function to download spaCy model
def download_spacy_model():
    """
    Download required spaCy model
    """
    print("\n===== Downloading spaCy Model =====")
    try:
        if is_package_installed('spacy'):
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
            print("spaCy model downloaded successfully!")
        else:
            print("spaCy not installed. Skipping model download.")
    except Exception as e:
        print(f"Error downloading spaCy model: {e}")
        return False
    
    return True

# Function to download datasets
def download_datasets():
    """
    Download required datasets
    """
    print("\n===== Downloading Datasets =====")
    download_script_path = os.path.join(DATA_DIR, 'download_datasets.py')
    
    if os.path.exists(download_script_path):
        try:
            subprocess.check_call([sys.executable, download_script_path])
            print("Datasets downloaded successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error downloading datasets: {e}")
            print("Continuing without downloading datasets...")
    else:
        print("download_datasets.py not found. Skipping dataset download.")
    
    return True

# Function to train models
def train_models():
    """
    Train machine learning models
    """
    print("\n===== Training Models =====")
    train_script_path = os.path.join(MODELS_DIR, 'train_simple_model.py')
    
    if os.path.exists(train_script_path):
        try:
            subprocess.check_call([sys.executable, train_script_path])
            print("Models trained successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error training models: {e}")
            return False
    else:
        print("train_simple_model.py not found. Skipping model training.")
    
    return True

# Function to run Streamlit app
def run_streamlit_app():
    """
    Run the Streamlit app
    """
    print("\n===== Running Streamlit App =====")
    app_script_path = os.path.join(APP_DIR, 'app.py')
    
    if os.path.exists(app_script_path):
        try:
            # Change to app directory to ensure relative paths work correctly
            os.chdir(APP_DIR)
            
            # Run Streamlit app
            print("Starting Streamlit app...")
            print("Access the app at http://localhost:8501")
            print("Press Ctrl+C to stop the app")
            
            # Run the Streamlit app
            subprocess.check_call([sys.executable, '-m', 'streamlit', 'run', app_script_path])
        except subprocess.CalledProcessError as e:
            print(f"Error running Streamlit app: {e}")
            return False
        except KeyboardInterrupt:
            print("\nStreamlit app stopped by user.")
    else:
        print("app.py not found. Cannot run Streamlit app.")
        return False
    
    return True

# Main function
def main():
    """
    Main function to run the entire project
    """
    parser = argparse.ArgumentParser(description='Fake News Detection Project')
    parser.add_argument('--skip-install', action='store_true', help='Skip installing requirements')
    parser.add_argument('--skip-nltk', action='store_true', help='Skip downloading NLTK data')
    parser.add_argument('--skip-spacy', action='store_true', help='Skip downloading spaCy model')
    parser.add_argument('--skip-datasets', action='store_true', help='Skip downloading datasets')
    parser.add_argument('--skip-training', action='store_true', help='Skip training models')
    parser.add_argument('--app-only', action='store_true', help='Only run the Streamlit app')
    
    args = parser.parse_args()
    
    print("===== Fake News Detection Project =====")
    print("This script will set up and run the entire project.")
    
    # If app-only flag is set, only run the Streamlit app
    if args.app_only:
        run_streamlit_app()
        return
    
    # Install requirements
    if not args.skip_install:
        if not install_requirements():
            print("Failed to install requirements. Exiting.")
            return
    
    # Download NLTK data
    if not args.skip_nltk:
        if not download_nltk_data():
            print("Failed to download NLTK data. Exiting.")
            return
    
    # Download spaCy model
    if not args.skip_spacy:
        if not download_spacy_model():
            print("Failed to download spaCy model. Continuing anyway...")
    
    # Download datasets
    if not args.skip_datasets:
        if not download_datasets():
            print("Failed to download datasets. Continuing anyway...")
    
    # Train models
    if not args.skip_training:
        if not train_models():
            print("Failed to train models. Exiting.")
            return
    
    # Run Streamlit app
    run_streamlit_app()

# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()