"""Data acquisition from UK STATS19, Kaggle US Accidents, and MoRTH."""
import sys
import os
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.config import (RAW_UK_STATS19, RAW_US_ACCIDENTS, RAW_MORTH, 
                            KAGGLE_USERNAME, KAGGLE_KEY, setup_logging, ensure_dirs,
                            get_config_section)

logger = setup_logging(__name__)

def download_uk_stats19():
    """Download UK STATS19 data for year 2023."""
    logger.info("Starting UK STATS19 data download.")
    urls = [
        "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2023.csv",
        "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-casualty-2023.csv",
        "https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-vehicle-2023.csv"
    ]
    
    os.makedirs(RAW_UK_STATS19, exist_ok=True)
    
    for url in urls:
        filename = url.split("/")[-1]
        filepath = Path(RAW_UK_STATS19) / filename
        
        if filepath.exists():
            logger.info(f"File {filename} already exists. Skipping download.")
            continue
            
        logger.info(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"Successfully downloaded {filename}.")
        except Exception as e:
            logger.error(f"Failed to download {filename}: {e}")

def download_us_accidents():
    """Download US Accidents dataset from Kaggle."""
    logger.info("Checking US Accidents data.")
    os.makedirs(RAW_US_ACCIDENTS, exist_ok=True)
    
    existing_files = list(Path(RAW_US_ACCIDENTS).glob("US_Accidents*.csv"))
    if existing_files:
        logger.info(f"US Accidents data already exists: {existing_files[0].name}")
        return
        
    if KAGGLE_USERNAME and KAGGLE_KEY:
        logger.info("Kaggle credentials found. Attempting download via Kaggle API.")
        os.environ['KAGGLE_USERNAME'] = KAGGLE_USERNAME
        os.environ['KAGGLE_KEY'] = KAGGLE_KEY
        
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'sobhanmoosavi/us-accidents', 
                path=str(RAW_US_ACCIDENTS), 
                unzip=True
            )
            logger.info("Successfully downloaded US Accidents data.")
        except Exception as e:
            logger.error(f"Failed to download via Kaggle API: {e}")
    else:
        logger.warning(
            "Kaggle credentials not found in environment (KAGGLE_USERNAME, KAGGLE_KEY).\n"
            "Please download the US Accidents dataset manually:\n"
            "1. Visit https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents\n"
            "2. Download the archive\n"
            "3. Extract the CSV file to data/raw/us_accidents/"
        )

def download_morth():
    """Download MoRTH dataset for Indian road accidents."""
    logger.info("Starting MoRTH data download.")
    os.makedirs(RAW_MORTH, exist_ok=True)
    
    logger.info(
        "Note: MoRTH data is aggregate-level (state-wise), not record-level.\n"
        "Attempting to search opencity.in CKAN API..."
    )
    
    search_url = "https://data.opencity.in/api/3/action/package_search?q=road+accidents+india"
    try:
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()
        
        # In a complete implementation, this would parse the search results and download
        # For now, we will print manual instructions as the structure can be highly variable
        if data.get('success'):
            logger.info("Successfully queried OpenCity API.")
        else:
            logger.warning("OpenCity API query returned unsuccessful status.")
            
        logger.warning(
            "Automatic extraction from OpenCity CKAN not fully implemented.\n"
            "Please download the MoRTH data manually if required:\n"
            "1. Visit relevant Indian government data portals or OpenCity.\n"
            "2. Download the state-wise aggregate dataset.\n"
            "3. Save to data/raw/morth/"
        )
    except Exception as e:
        logger.error(f"Failed to query MoRTH data: {e}")
        logger.warning(
            "Please download the MoRTH data manually and save to data/raw/morth/"
        )

def main():
    ensure_dirs()
    logger.info("Starting data acquisition process.")
    
    download_uk_stats19()
    download_us_accidents()
    download_morth()
    
    logger.info("Data acquisition process completed.")

if __name__ == '__main__':
    main()
