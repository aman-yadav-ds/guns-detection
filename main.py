import io
import os
from fastapi import FastAPI, File, UploadFile #For uploading image
from fastapi.responses import StreamingResponse
import torch
from torchvision import models, transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image, ImageDraw
from config.config import *

num_classes = 2  # Example: background + 1 class (change to your number of classes)
model = models.detection.fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(os.path.join(MODEL_PATH, "fastercnn.pth"), map_location=torch.device("cpu")))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = transforms.Compose([
    transforms.ToTensor(),
])

app = FastAPI()

def predict_and_draw(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)
        
    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    for box, score in zip(boxes, scores):
        if score>0.7:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=2)

    return img_rgb

@app.get("/")
def read_root():
    return {"message": "Welcome to the Guns Object Detection API."}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    output_image = predict_and_draw(image)

    img_byte_array = io.BytesIO()
    output_image.save(img_byte_array, format="PNG")
    img_byte_array.seek(0)

    return StreamingResponse(img_byte_array, media_type="image/png")