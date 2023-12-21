import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torchvision.models.feature_extraction import create_feature_extractor



class CustomCosineEmbeddingLoss(nn.Module):
    """
    Custom implementation of Cosine Embedding Loss with margin.

    Parameters:
    - margin (float): Margin value for the loss computation.
    """

    def __init__(self, margin=0.2):
        super(CustomCosineEmbeddingLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, y):
        """
        Forward pass of the loss computation.

        Args:
        - x1 (torch.Tensor): Tensor representing the first set of embeddings.
        - x2 (torch.Tensor): Tensor representing the second set of embeddings.
        - y (torch.Tensor): Tensor representing the binary labels (1 for positive pairs, -1 for negative pairs).

        Returns:
        - loss (torch.Tensor): Computed Cosine Embedding Loss.
        """

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(x1, x2, dim=1, eps=1e-6)

        # Loss computation based on the provided formula
        loss = torch.where(y == 1, 1 - cosine_similarity, torch.max(torch.zeros_like(cosine_similarity), cosine_similarity - self.margin))

        # Take the mean over the batch
        loss = torch.mean(loss)

        return loss


class MultimodalModel(nn.Module):
    """
    Multimodal Model combining sentence and image embeddings.

    Parameters:
    - sentence_embedding_model: Sentence embedding model.
    - image_embedding_model: Image embedding model.
    """

    def __init__(self, sentence_embedding_model, image_embedding_model):
        super(MultimodalModel, self).__init__()
        self.sentence_embedding_model = sentence_embedding_model
        self.image_embedding_model = image_embedding_model
        self.fc = nn.Linear(2, 1)

    def forward(self, image_data1, image_data2, text_data1, text_data2, call_type='train'):
        """
        Forward pass of the multimodal model.

        Args:
        - image_data1 (torch.Tensor): Tensor representing the first set of images.
        - image_data2 (torch.Tensor): Tensor representing the second set of images.
        - text_data1 (list): List of text data for the first set.
        - text_data2 (list): List of text data for the second set.

        Returns:
        - fused_output (torch.Tensor): Output of the multimodal model.
        - sentence_embeddings1 (torch.Tensor): Embeddings from the first set of sentences.
        - sentence_embeddings2 (torch.Tensor): Embeddings from the second set of sentences.
        - image_embeddings1 (torch.Tensor): Embeddings from the first set of images.
        - image_embeddings2 (torch.Tensor): Embeddings from the second set of images.
        """

        # Encode sentences to obtain embeddings
        sentence_embeddings1 = self.sentence_embedding_model.encode(text_data1, convert_to_tensor=True)
        sentence_embeddings2 = self.sentence_embedding_model.encode(text_data2, convert_to_tensor=True)

        # Create a feature extractor for images
        return_nodes = {'flatten': 'flatten'}
        feature_extractor = create_feature_extractor(self.image_embedding_model, return_nodes=return_nodes)

        # Encode images to obtain embeddings
        image_embeddings1 = feature_extractor(image_data1)['flatten']
        image_embeddings2 = feature_extractor(image_data2)['flatten']

        if call_type == 'inference':
            sentence_embeddings1, sentence_embeddings1 = sentence_embeddings1.unsqueeze(0), sentence_embeddings1.unsqueeze(0)

        # Compute cosine similarities
        cosine1 = F.cosine_similarity(image_embeddings1, image_embeddings2, dim=1)
        cosine2 = F.cosine_similarity(sentence_embeddings1, sentence_embeddings2, dim=-1)

        # Concatenate cosine similarities


        if call_type == 'inference':
            cosine_features = torch.cat([cosine1.unsqueeze(0), cosine2.unsqueeze(0)], dim=1)
        else:
            cosine_features = torch.cat([cosine1.unsqueeze(1), cosine2.unsqueeze(1)], dim=1)

        fused_output = self.fc(cosine_features)


        # Return outputs and embeddings
        return fused_output.squeeze(1), sentence_embeddings1, sentence_embeddings2, image_embeddings1, image_embeddings2
