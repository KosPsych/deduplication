from sentence_transformers import SentenceTransformer
from model import MultimodalModel
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.models as models
import pandas as pd
from dataset import *
from torch.utils.data import DataLoader
import numpy as np
from flask import Flask, jsonify





def test():
    """
    Perform evaluation on the test dataset using a pre-trained multimodal model.

    Returns:
    - JSON response containing evaluation metrics.
    """
    # Import necessary constants
    
    from utils.constants import DATA_PATH, MODEL_PATH

    # Load the test dataset and define column names
    dataset = pd.read_csv(DATA_PATH + "data.csv")
    text_column1_name = 'title'
    text_column2_name = 'title_2'
    image_id_column1_name = 'img_identifier'
    image_id_column2_name = 'img_identifier_2'
    label_name = 'label'
    image_folder = DATA_PATH + 'images'
    dataset_size = len(dataset)

    # Create a custom dataset and dataloader
    test_dataset = CustomDataset('test', dataset_size, dataset, image_folder, text_column1_name, text_column2_name, image_id_column1_name, image_id_column2_name, label_name)
    batch_size = 32
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Initialize the device and load pre-trained models
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    resnet18 = models.resnet18(pretrained=True)
    sentence_transformer = SentenceTransformer('all-MiniLM-L12-v2').to(device)
    model = MultimodalModel(sentence_transformer, resnet18).to(device)

    model.load_state_dict(torch.load(MODEL_PATH + 'model.pth'))

    # Evaluation loop
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in test_dataloader:


            text_data1, text_data2, image_data1, image_data2, labels = batch['text_data1'], batch['text_data2'], batch['image_data1'], batch['image_data2'], batch['labels']

            # Move tensors to the device
            image_data1, image_data2 = image_data1.to(device), image_data2.to(device)

            outputs, _, _, _, _ = model(image_data1, image_data2, text_data1, text_data2)

            predictions.extend(outputs.squeeze().cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Convert predictions to binary (0 or 1) based on a threshold
    threshold = 0.5
    binary_predictions = np.array(predictions) > threshold
    binary_predictions = binary_predictions.astype(int)
    # Calculate evaluation metrics

    accuracy = accuracy_score(true_labels, binary_predictions)
    precision = precision_score(true_labels, binary_predictions)
    recall = recall_score(true_labels, binary_predictions)
    f1 = f1_score(true_labels, binary_predictions)

    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    })
