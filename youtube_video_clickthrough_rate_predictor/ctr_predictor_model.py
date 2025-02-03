import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import numpy as np
import gradio as gr
import shap
import cv2

# Constants
YT_TN_DIR = "yt_thumbnail" #This directory has all thumbnails
YT_STAT_DIR = 'yt_youtube' #This directory has youtube statistics data
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Load Data
class CTRDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_name = self.df.iloc[idx]['image_name']
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        ctr = self.df.iloc[idx]['ctr']

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(ctr, dtype=torch.float32)

# Load dataframe
df_ml = pd.read_csv(os.path.join(YT_STAT_DIR, "yt_ml.csv"))

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = CTRDataset(df_ml, YT_TN_DIR, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define Model
class CTRPredictor(nn.Module):
    def __init__(self):
        super(CTRPredictor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
    
    def forward(self, x):
        return self.backbone(x)

# Train Model
model = CTRPredictor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    model.train()
    for images, ctr in train_loader:
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, ctr)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item()}")

# Save Model
torch.save(model.state_dict(), "ctr_predictor.pth")

# SHAP Explanation
explainer = shap.GradientExplainer(model, torch.stack([train_dataset[i][0] for i in range(5)]))

def explain_ctr(image):
    model.load_state_dict(torch.load("ctr_predictor.pth"))
    model.eval()
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        ctr_pred = model(image_tensor).item()
    shap_values = explainer.shap_values(image_tensor)
    shap_img = np.abs(shap_values[0]).sum(axis=0)
    shap_img = (shap_img - shap_img.min()) / (shap_img.max() - shap_img.min())
    shap_img = np.uint8(255 * shap_img)
    shap_img = cv2.applyColorMap(shap_img, cv2.COLORMAP_JET)
    return ctr_pred, Image.fromarray(shap_img), "Highlighted areas influence the CTR prediction."

# Gradio UI
iface = gr.Interface(
    fn=explain_ctr,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Number(label="Predicted CTR"),
        gr.Image(label="SHAP Explanation"),
        gr.Textbox(label="Explanation"),
    ],
    title="CTR Predictor with SHAP",
    description="Upload an image to predict CTR and visualize influential regions using SHAP.",
)

iface.launch()
