import copy

class EarlyStopper:
    """
    Implements early stopping for training neural networks.
    Monitors a validation metric to stop training if no improvement.

    Attributes:
    - patience (int): Specified patience value.
    - min_delta (float): Specified min_delta value.
    - counter (int): Counts epochs with no improvement.
    - min_validation_loss (float): Minimum validation loss. Initialized to infinity.
    - best_model_state: The best model found so far (lowest validation loss)

    Methods:
    - early_stop(validation_loss): Returns True if stopping criteria are met; otherwise, False.
    """

    def __init__(self, patience=2, min_delta=0):
        """Initializes an EarlyStopper instance."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.best_model_state = None

    def early_stop(self, validation_loss, model_state):
        """Returns True if stopping criteria are met; otherwise, False."""
        if validation_loss < self.min_validation_loss:

            self.min_validation_loss = validation_loss
            self.counter = 0
            self.best_model_state = copy.deepcopy(model_state)
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def training_step( model,  data_loader, bce_loss, cos_loss, optimizer, device):
    """
    Performs a single training step
    Args:
    model: The multimodal model
    data_loader: PyTorch dataloader, containing (images, images).
    bce_loss: PyTorch bce_loss, computes loss between 2 images.
    cos_loss: PyTorch cos_loss, computes loss between embeddings.
    optimizer: PyTorch optimizer.
    device: "cuda" or "cpu"
    Returns: Train Loss
    """

    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for batch in data_loader:

        text_data1, text_data2, image_data1, image_data2, labels= (batch['text_data1'],
                                                                  batch['text_data2'],
                                                                  batch['image_data1'],
                                                                  batch['image_data2'],
                                                                  batch['labels'])

        optimizer.zero_grad()

        image_data1, image_data2 = image_data1.to(device), image_data2.to(device)
        labels = labels.to(device)

        # Get output of the model
        outputs, sentence_embeddings1, sentence_embeddings2, image_embeddings1, image_embeddings2 = model(image_data1, image_data2, text_data1, text_data2)

        # Get tensors to device (GPU)
        outputs, sentence_embeddings1, sentence_embeddings2, image_embeddings1, image_embeddings2 =     (outputs.to(device),
                                                                                                        sentence_embeddings1.to(device),
                                                                                                        sentence_embeddings2.to(device),
                                                                                                        image_embeddings1.to(device), image_embeddings2.to(device))

        loss = (bce_loss(outputs, labels.float())  # BCE loss
               + cos_loss(sentence_embeddings1, sentence_embeddings2, labels.float()) # cosine embedding loss for sentence embeddings
               + cos_loss(image_embeddings1, image_embeddings2, labels.float())) # cosine embedding loss for image embeddings

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / num_batches



def validation_step( model,  data_loader, bce_loss, cos_loss,  device):
            """
            Performs a single validation step
            Args:
            model: The multimodal model
            data_loader: PyTorch dataloader, containing (images, images).
            bce_loss: PyTorch bce_loss, computes loss between 2 images.
            cos_loss: PyTorch cos_loss, computes loss between embeddings.
            device: "cuda" or "cpu"
            Returns: Train Loss
            """

            model.eval()
            total_loss = 0.0
            num_batches = len(data_loader)
            for batch in data_loader:

                text_data1, text_data2, image_data1, image_data2, labels= (batch['text_data1'],
                                                                        batch['text_data2'],
                                                                        batch['image_data1'],
                                                                        batch['image_data2'],
                                                                        batch['labels'])


                image_data1, image_data2 = image_data1.to(device), image_data2.to(device)
                labels = labels.to(device)
                # Get image embeddings
                outputs, sentence_embeddings1, sentence_embeddings2, image_embeddings1, image_embeddings2 = model(image_data1, image_data2, text_data1, text_data2)


                 # Get tensors to device (GPU)
                outputs, sentence_embeddings1, sentence_embeddings2, image_embeddings1, image_embeddings2 = (outputs.to(device),
                                                                                                        sentence_embeddings1.to(device),
                                                                                                        sentence_embeddings2.to(device),
                                                                                                        image_embeddings1.to(device), image_embeddings2.to(device))

                loss = (bce_loss(outputs, labels.float())  # BCE loss
                        + cos_loss(sentence_embeddings1, sentence_embeddings2, labels.float()) # cosine embedding loss for sentence embeddings
                        + cos_loss(image_embeddings1, image_embeddings2, labels.float())) # cosine embedding loss for image embeddings


                total_loss += loss.item()


            return total_loss / num_batches
