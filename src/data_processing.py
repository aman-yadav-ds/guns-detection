import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from src.logger import get_logger
from src.custom_exception import CustomException

logger = get_logger(__name__)

class GunDataset(Dataset):

    def __init__(self, root_dir, device:str="cpu"):
        self.img_path = os.path.join(root_dir, "Images")
        self.labels_path = os.path.join(root_dir, "Labels")
        self.device = device

        self.img_name = sorted(os.listdir(self.img_path))
        self.label_name = sorted(os.listdir(self.labels_path))

        logger.info("Data Processing Initialized...")
    
    def __getitem__(self, idx):
        try:
            logger.info(f"Loading Data for Index {idx}")
            #### Loading Images
            img_path = os.path.join(self.img_path, str(self.img_name[idx]))

            logger.info(f"Image Path : {img_path}")
            image = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

            img_res = img_rgb / 255 # Normalizing the image.
            img_res = torch.as_tensor(img_res).permute(2, 0, 1)

            #### Loading Labels

            label_name = self.img_name[idx].rsplit('.', 1)[0] + ".txt"
            label_path = os.path.join(self.labels_path, str(label_name))

            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not Found in {label_path}")

            target = {
                "boxes": torch.tensor([]),
                "area": torch.tensor([]),
                "img_id": torch.tensor([idx]),
                "labels": torch.tensor([], dtype=torch.int64)
            }

            with open(label_path, "r") as label_file:
                l_count = int(label_file.readline())
                box = [list(map(int, label_file.readline().split())) for _ in range(l_count)]

            if box:
                area = [(b[2] - b[0]) * (b[3] - b[1]) for b in box]
            
                labels = [1] * len(box)

                target["boxes"] = torch.tensor(box, dtype=torch.float32)
                target["area"] = torch.tensor(area, dtype=torch.float32)
                target["labels"] = torch.tensor(labels, dtype=torch.int64)

            img_res = img_res.to(self.device)
            for key in target:
                target[key] = target[key].to(self.device)
            return img_res, target
        except Exception as e:
            logger.error(f"Error while loading data {e}")
            raise CustomException("Failed to load data", e)

    def __len__(self):
        return len(self.img_name)
    
if __name__ == "__main__":
    root_path = "artifacts/data/raw"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = GunDataset(root_dir=root_path, device=device)

    image, target = dataset[0]

    print("Image Shape: ", image.shape)
    print("Target Keys: ", target.keys())
    print("Bounding Boxes: ", target["boxes"])