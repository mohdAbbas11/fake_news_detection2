import os
import pandas as pd
import requests
import zipfile
import io
import kaggle
import json
import shutil
from tqdm import tqdm

# Create necessary directories
data_dir = os.path.abspath(os.path.dirname(__file__))
raw_dir = os.path.join(data_dir, 'raw')
processed_dir = os.path.join(data_dir, 'processed')

for directory in [raw_dir, processed_dir]:
    os.makedirs(directory, exist_ok=True)

# Function to download file from URL
def download_file(url, save_path):
    """
    Download a file from a URL and save it to the specified path
    
    Args:
        url (str): URL to download from
        save_path (str): Path to save the file to
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Get file size for progress bar
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Download with progress bar
        with open(save_path, 'wb') as f, tqdm(
            desc=os.path.basename(save_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                bar.update(size)
        
        print(f"Downloaded {os.path.basename(save_path)}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

# Function to download and extract zip file
def download_and_extract_zip(url, extract_dir):
    """
    Download a zip file from a URL and extract it to the specified directory
    
    Args:
        url (str): URL to download from
        extract_dir (str): Directory to extract the zip file to
    """
    try:
        print(f"Downloading and extracting from {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        # Create directory if it doesn't exist
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(extract_dir)
        
        print(f"Extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"Error downloading or extracting {url}: {e}")
        return False

# Function to download dataset from Kaggle
def download_kaggle_dataset(dataset, path):
    """
    Download a dataset from Kaggle
    
    Args:
        dataset (str): Kaggle dataset identifier (e.g., 'clmentbisaillon/fake-and-real-news-dataset')
        path (str): Path to save the dataset to
    """
    try:
        # Check if kaggle.json exists
        kaggle_dir = os.path.expanduser('~/.kaggle')
        kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
        
        if not os.path.exists(kaggle_json):
            print("Kaggle API credentials not found. Please set up your Kaggle API credentials.")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click on 'Create New API Token'")
            print("3. Save the kaggle.json file to ~/.kaggle/kaggle.json")
            print("4. Run this script again")
            return False
        
        # Download dataset
        print(f"Downloading Kaggle dataset: {dataset}")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset, path=path, unzip=True, quiet=False)
        print(f"Downloaded Kaggle dataset: {dataset}")
        return True
    except Exception as e:
        print(f"Error downloading Kaggle dataset {dataset}: {e}")
        return False

# Main function to download all datasets
def download_all_datasets():
    """
    Download all datasets for the fake news detection project
    """
    print("Starting dataset downloads...")
    
    # 1. Download Indian Fake News Dataset
    indian_fake_news_url = "https://raw.githubusercontent.com/shivangi-aneja/SAFE/master/data/SAFE_dataset.csv"
    indian_fake_news_path = os.path.join(raw_dir, "indian_fake_news.csv")
    download_file(indian_fake_news_url, indian_fake_news_path)
    
    # 2. Download Kaggle's Fake and Real News Dataset
    kaggle_dataset = "clmentbisaillon/fake-and-real-news-dataset"
    download_kaggle_dataset(kaggle_dataset, raw_dir)
    
    # 3. Download FakeNewsNet Dataset (sample)
    fakenewsnet_url = "https://github.com/KaiDMML/FakeNewsNet/raw/master/dataset/sample.zip"
    fakenewsnet_dir = os.path.join(raw_dir, "fakenewsnet")
    download_and_extract_zip(fakenewsnet_url, fakenewsnet_dir)
    
    print("All datasets downloaded successfully!")

# Run the download function if this script is executed directly
if __name__ == "__main__":
    download_all_datasets()