import os
import kagglehub
import shutil
import zipfile

from src.logger import get_logger
from src.custom_exception import CustomException
from config.config import DATASET_NAME, DATA_DIR

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, dataset_name:str, data_dir:str):
        try:    
            self.dataset_name = dataset_name
            self.data_dir = data_dir

            self.raw_dir = os.path.join(self.data_dir, "raw")
            os.makedirs(self.raw_dir, exist_ok=True)

            logger.info("Data Ingestion Initialized.")
        except Exception as e:
            logger.error("Error while initializing Ingestion object...")
            raise CustomException("Failed to Initialize ingestor object.", e)
        
    
    def extract_data(self, path:str):
        try:
            if path.endswith('.zip'):
                logger.info("Extracting Zip files...")

                with zipfile.ZipFile(path, 'r') as zip_ref:
                    zip_ref.extractall(path)

            images_folder = os.path.join(path, "Images")
            labels_folder = os.path.join(path, "Labels")

            if os.path.exists(images_folder):
                shutil.move(images_folder, os.path.join(self.raw_dir, "Images"))
                logger.info("Images moved Successfully.")
            else:
                logger.info("Images Folder don't exist.")

            
            if os.path.exists(labels_folder):
                shutil.move(labels_folder, os.path.join(self.raw_dir, "Labels"))
                logger.info("Labels moved Successfully.")
            else:
                logger.info("Labels Folder don't exist.")
        except Exception as e:
            logger.error("Error while extracting...")
            raise CustomException("Failed to Extract data.", e)
        
    def download_data(self):
        try:
            path = kagglehub.dataset_download(self.dataset_name)
            logger.info(f"Downloaded the data from {path}")

            self.extract_data(path)
        except Exception as e:
            logger.error("Error while downloading data...")
            raise CustomException("Failed to download data.", e)


if __name__=="__main__":
    ingestor = DataIngestion(DATASET_NAME, DATA_DIR)

    ingestor.download_data()