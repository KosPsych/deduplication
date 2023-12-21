from sentence_transformers import SentenceTransformer
from model import MultimodalModel
import torch
import torchvision.models as models
import pandas as pd
import numpy as np
from flask import jsonify
from torchvision import  transforms
import torch.nn.functional as F



def inference(image1, image2, title1, title2):
    """
    Perform multimodal inference using a pretrained model.

    Args:
    - image1 (PIL.Image): Image data for the first set of images.
    - image2 (PIL.Image): Image data for the second set of images.
    - title1 (str): A parameter indicating the intensity of the style transfer procedure for the first set.
    - title2 (str): A parameter indicating the intensity of the style transfer procedure for the second set.

    Returns:
    - JSON response containing scores.
    """

    # Check if GPU is available, else use CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    from utils.constants import MODEL_PATH

    # Define a transformation to apply to the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Preprocess images for inference
    image1 = transform(image1).to(device).unsqueeze(0)
    image2 = transform(image2).to(device).unsqueeze(0)

    # Load pre-trained models
    resnet18 = models.resnet18(pretrained=True)
    sentence_transformer = SentenceTransformer('all-MiniLM-L12-v2').to(device)
    model = MultimodalModel(sentence_transformer, resnet18).to(device)

    # Load the pre-trained weights for the model
    model.load_state_dict(torch.load(MODEL_PATH + 'model.pth'))

    # Perform inference on the model
    outputs, _, _, _, _ = model(image1, image2, title1, title2, 'inference')

    # Apply sigmoid activation function for binary classification
    outputs = F.sigmoid(outputs)

    # Convert the output to a JSON response
    return jsonify({'score/s': outputs.detach().cpu().numpy().tolist()})
