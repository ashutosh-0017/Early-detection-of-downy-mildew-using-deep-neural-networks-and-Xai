import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import json
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

def remove_background_grabcut(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    h, w = img_bgr.shape[:2]
    margin = int(min(h, w) * 0.05)
    rect = (margin, margin, w - 2*margin, h - 2*margin)
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    white_bg = np.ones_like(img_rgb) * 255
    result = np.where(mask2[:, :, np.newaxis] == 0, white_bg, img_rgb)
    return result.astype(np.uint8)

# Configure Streamlit page
st.set_page_config(page_title="MildewGuard AI", layout="centered", page_icon="🌿")

st.title("🌿 MildewGuard AI")
st.markdown("Upload a vineyard leaf image to detect **Downy Mildew**.")
st.markdown("---")

# 1. Load Model and Info
@st.cache_resource
def load_model_and_info():
    if not os.path.exists('best_model_info.json') or not os.path.exists('best_model.pth'):
        return None, None
        
    with open('best_model_info.json', 'r') as f:
        info = json.load(f)
        
    arch = info['architecture']
    if arch == "ResNet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif arch == "EfficientNetB0":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif arch == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        
    # Load weights
    model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
    model.eval()
    return model, info

model, info = load_model_and_info()

if model is None:
    st.error("Model files not found! Please run the notebook and train the models first.")
    st.stop()

# 2. Image Upload
uploaded_file = st.file_uploader("Upload Image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Transform for model
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img_resized).unsqueeze(0)
    
    # 3. Prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_class = torch.max(probabilities, 0)
        
    class_names = info['class_names']
    pred_name = class_names[predicted_class.item()]
    conf_pct = confidence.item() * 100
    
    st.subheader(f"🔍 Prediction: **{pred_name.title()}**")
    st.write(f"**Confidence:** {conf_pct:.2f}%")
    st.markdown("---")
    
    # 4. Grad-CAM Explanation
    st.subheader("🧠 AI Explanation (Grad-CAM)")
    
    arch = info['architecture']
    if arch == "ResNet50":
        target_layers = [model.layer4[-1]]
    elif arch == "EfficientNetB0":
        target_layers = [model.features[-1]]
    elif arch == "MobileNetV2":
        target_layers = [model.features[-1]]
        
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class.item())]
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    rgb_img = img_resized / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_resized, caption="Original Image", use_container_width=True)
    with col2:
        st.image(visualization, caption="Grad-CAM Heatmap", use_container_width=True)
