import torch
from torch import nn
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np
import cv2

classes = ["Норма", "Пневмонія"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

target_layer = model.layer4[-1]
cam = GradCAM(model=model, target_layers=[target_layer])

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_with_cam(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    confidence, pred = torch.max(probs, 0)

    grayscale_cam = cam(
    	input_tensor=img,
    	targets=[ClassifierOutputTarget(pred.item())]
    )[0]

    rgb_img = np.array(image.resize((224, 224))).astype(np.float32) / 255.0

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return classes[pred], float(confidence), visualization