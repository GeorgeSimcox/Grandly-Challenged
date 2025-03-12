import torch
import timm
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import io
import os
import base64

# Define necessary configurations
class Config:
    backbone = "resnet18"
    n_classes = 8
    image_size = 200
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean = [0.485, 0.456, 0.406]  # Standard ImageNet mean
    std = [0.229, 0.224, 0.225]   # Standard ImageNet std

cfg = Config()

# Class names for predictions
CLASS_NAMES = [
    "1 polyethylene (PET)",
    "2 high density polyethylene (HDPE/PEHD)",
    "3 polyvinyl chloride (PVC)",
    "4 low density polyethylene (LDPE)",
    "5 polypropylene (PP)",
    "6 polystyrene (PS)",
    "7 other resins",
    "8 no plastic"
]

# Plastic descriptions and recycling information
PLASTIC_INFO = {
    "1 polyethylene (PET)": {
        "common_uses": "Water bottles, soda bottles, food containers",
        "recyclable": "Highly recyclable",
        "disposal": "Rinse and place in recycling bin",
        "environmental_impact": "One of the most widely recycled plastics"
    },
    "2 high density polyethylene (HDPE/PEHD)": {
        "common_uses": "Milk jugs, detergent bottles, toys",
        "recyclable": "Highly recyclable",
        "disposal": "Rinse and place in recycling bin",
        "environmental_impact": "Lower carbon footprint than other plastics"
    },
    "3 polyvinyl chloride (PVC)": {
        "common_uses": "Pipes, medical equipment, window frames",
        "recyclable": "Difficult to recycle",
        "disposal": "Check local guidelines, often not accepted in curbside recycling",
        "environmental_impact": "Can release toxic chemicals if incinerated"
    },
    "4 low density polyethylene (LDPE)": {
        "common_uses": "Plastic bags, shrink wraps, squeeze bottles",
        "recyclable": "Recyclable but not always accepted in curbside programs",
        "disposal": "Check local guidelines, often accepted at grocery store drop-offs",
        "environmental_impact": "Commonly found in ocean pollution"
    },
    "5 polypropylene (PP)": {
        "common_uses": "Yogurt containers, bottle caps, straws",
        "recyclable": "Increasingly recyclable",
        "disposal": "Rinse and place in recycling bin",
        "environmental_impact": "Requires less energy to produce than PET or PVC"
    },
    "6 polystyrene (PS)": {
        "common_uses": "Styrofoam, disposable cups, packaging",
        "recyclable": "Rarely recyclable in curbside programs",
        "disposal": "Check for specialized recycling programs",
        "environmental_impact": "Slow to degrade and problematic in the environment"
    },
    "7 other resins": {
        "common_uses": "Various products including multi-layer packaging",
        "recyclable": "Generally difficult to recycle",
        "disposal": "Check local guidelines",
        "environmental_impact": "Often ends up in landfills"
    },
    "8 no plastic": {
        "common_uses": "N/A",
        "recyclable": "N/A",
        "disposal": "Dispose according to material type",
        "environmental_impact": "N/A"
    }
}

def load_model(model_path):
    """Load the trained model from a saved state_dict"""
    model = timm.create_model(cfg.backbone, pretrained=False, num_classes=cfg.n_classes)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.to(cfg.device)
    model.eval()
    return model

def preprocess_image_from_bytes(image_bytes):
    """Preprocess the image from bytes to match the model's expected input"""
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std),
    ])

    # Convert bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    
    # Apply transformation
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.to(cfg.device)

def preprocess_image_from_base64(base64_string):
    """Preprocess the image from base64 string to match the model's expected input"""
    # Remove the data URL prefix if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
        
    # Decode base64 to bytes
    image_bytes = base64.b64decode(base64_string)
    return preprocess_image_from_bytes(image_bytes)

def predict(image_input, model_path, is_base64=False):
    """
    Predict the class of a given image using the trained model
    
    Parameters:
    - image_input: Either base64 string (if is_base64=True) or bytes (if is_base64=False)
    - model_path: Path to the model file
    - is_base64: Whether the input is a base64 string
    
    Returns:
    - Dictionary with prediction results
    """
    model = load_model(model_path)
    
    if is_base64:
        image = preprocess_image_from_base64(image_input)
    else:
        image = preprocess_image_from_bytes(image_input)

    with torch.no_grad():
        output = model(image)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        
        # Get all probabilities for display
        all_probs = probabilities[0].cpu().numpy()
        
        # Get top prediction
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item() * 100
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        # Get top 3 predictions
        top_k_values, top_k_indices = torch.topk(probabilities, 3, dim=1)
        top_3_predictions = []
        
        for i in range(3):
            idx = top_k_indices[0, i].item()
            prob = top_k_values[0, i].item() * 100
            top_3_predictions.append({
                "class": CLASS_NAMES[idx],
                "confidence": float(prob),
                "info": PLASTIC_INFO[CLASS_NAMES[idx]]
            })

    return {
        "top_prediction": {
            "class": predicted_class,
            "confidence": float(confidence),
            "info": PLASTIC_INFO[predicted_class]
        },
        "top_3_predictions": top_3_predictions,
        "all_probabilities": [float(p * 100) for p in all_probs]
    }