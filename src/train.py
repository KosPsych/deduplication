import pandas as pd
from torch.utils.data import DataLoader
import numpy as np
from utils.raw_preprocessing import *
from utils.training_utils import *
from dataset import *
from model import *
from sentence_transformers import SentenceTransformer, util, losses
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torchvision.models as models



def train():
    """
    Train a multimodal model on a dataset using a combination of binary cross-entropy and cosine embedding loss.

    Returns:
    - Success message after training completion.
    """
    # Import necessary constants
    from utils.constants import DATA_PATH, MODEL_PATH

    # Load the dataset and set the device
    dataset = pd.read_csv(DATA_PATH + "data.csv")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Example usage:
    text_column1_name = 'title'
    text_column2_name = 'title_2'
    image_id_column1_name = 'img_identifier'
    image_id_column2_name = 'img_identifier_2'
    label_name = 'label'
    image_folder = DATA_PATH + 'images'
    dataset_size = len(dataset)

    # Define transformations

    # Create custom datasets and DataLoaders for training and validation
    train_dataset = CustomDataset('train', dataset_size, dataset, image_folder, text_column1_name, text_column2_name, image_id_column1_name, image_id_column2_name, label_name)
    val_dataset = CustomDataset('val', dataset_size, dataset, image_folder, text_column1_name, text_column2_name, image_id_column1_name, image_id_column2_name, label_name)

    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    resnet18 = models.resnet18(pretrained=True)
    sentence_transformer = SentenceTransformer('all-MiniLM-L12-v2').to(device)
    model = MultimodalModel(sentence_transformer, resnet18).to(device)

    # Loss function and optimizer
    cos_loss = nn.CosineEmbeddingLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stopping = EarlyStopper()

    # Training loop
    for epoch in range(25):

        # Perform training step
        training_loss = training_step(model, train_dataloader, bce_loss, cos_loss, optimizer, device)

        with torch.no_grad():
            # Perform validation step
            validation_loss = validation_step(model, val_dataloader, bce_loss, cos_loss, device)

            # Check for early stopping criteria
            if early_stopping.early_stop(validation_loss, model.state_dict()) or epoch == 24:
                torch.save(early_stopping.best_model_state, MODEL_PATH + 'model.pth')
                break

        print("Epoch {}, training loss: {:.4f}, validation loss: {:.4f}".format(epoch + 1, training_loss, validation_loss))

    return 'successful training'
