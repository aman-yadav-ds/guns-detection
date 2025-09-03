import os
import sys
import time
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from src.model_architecture import FasterRCNNModel
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_processing import GunDataset
from config.config import *

logger = get_logger(__name__)

os.makedirs(MODEL_PATH, exist_ok=True)

class ModelTraining:
    def __init__(self, model_class, num_classes, lr, epochs, dataset_path, device):
        self.model_class = model_class
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.dataset_path = dataset_path
        self.device = device

        ##### Tensorboard
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = f"tensorboard_logs/{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        try:
            self.model = self.model_class(self.num_classes, self.device).model
            self.model.to(self.device)
            logger.info(f"Model Training initialized in device: {self.device}")

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
            logger.info(f"{self.optimizer} Optimizer has been initialized...")

        except Exception as e:
            logger.error("Failed to initialize model training.", e)
            raise CustomException("Error while initializing model training.", sys)

    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    def split_dataset(self):
        try:
            dataset = GunDataset(self.dataset_path, self.device)

            train_size = int(0.8*len(dataset))
            val_size = int(len(dataset) - train_size)
            
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=0, collate_fn=self.collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=3, shuffle=False, num_workers=0, collate_fn=self.collate_fn)

            logger.info("Data Splitted Successfully...")

            return train_loader, val_loader
        
        except Exception as e:
            logger.error("Failed to split dataset.", e)
            raise CustomException("Error while splitting dataset.", sys)
        
    def train(self):
        try:
            train_loader, val_loader = self.split_dataset()

            for epoch in range(self.epochs):
                logger.info(f"Starting Epoch {epoch}")

                self.model.train()

                for i, (images, targets) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    losses = self.model(images, targets)

                    if isinstance(losses, dict):
                        total_loss = 0
                        for key, val in losses.items():
                            if isinstance(val, torch.Tensor):
                                total_loss += val

                        if  total_loss == 0:
                            logger.error(f"There was error when capturing losses.")
                            raise ValueError("Total Loss Value is 0")
                        
                        self.writer.add_scalar("Loss/train", total_loss.item(), epoch*len(train_loader)+i)
                    
                    else:
                        total_loss = losses[0]
                        self.writer.add_scalar("Loss/train", total_loss.item(), epoch*len(train_loader)+i)

                    total_loss.backward()
                    self.optimizer.step()

                self.writer.flush()

                self.model.eval()
                with torch.no_grad():
                    for images, targets in val_loader:
                        val_losses = self.model(images, targets)
                        logger.info(type(val_losses))
                        logger.info(f"Validation Loss : {val_losses}")

                model_path = os.path.join(MODEL_PATH, "fastercnn.pth")
                torch.save(self.model.state_dict(), model_path)

                logger.info(f"Model Saved Successfully.")
        except Exception as e:
            logger.error("Failed to Train Model.", e)
            raise CustomException("Error while Train Model.", sys)
        
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = ModelTraining(model_class=FasterRCNNModel, 
                            num_classes=2, 
                            lr=0.0005,
                            epochs=15,
                            dataset_path=DATASET_PATH, 
                            device=device)
    trainer.train()