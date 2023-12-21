from torch.utils.data import Dataset
import torch
from PIL import Image
from torchvision import  transforms

class CustomDataset(Dataset):
    """
    Custom dataset class for multimodal data (text and images).
    """

    def __init__(self, split, dataset_size, dataframe, image_folder, text_column1, text_column2, image_id_column1, image_id_column2, label_name):
        """
        CustomDataset constructor.

        Args:
        - split: a string variable indicating the type of data (train, test or validation)
        - dataset_size: the number of samples i nthe dataframe
        - dataframe (pd.DataFrame): The input DataFrame containing text, image IDs, and labels.
        - image_folder (str): Path to the folder containing images.
        - text_column1 (str): Name of the column containing text data for the first modality.
        - text_column2 (str): Name of the column containing text data for the second modality.
        - image_id_column1 (str): Name of the column containing image IDs for the first modality.
        - image_id_column2 (str): Name of the column containing image IDs for the second modality.
        - label_name (str): Name of the column containing labels.
        """
        if split == 'train':
            self.dataframe = dataframe[:int(0.7*dataset_size)]
        elif split == 'val':
            self.dataframe = dataframe[int(0.7*dataset_size):int(0.9*dataset_size)]
        else:
            self.dataframe = dataframe[int(0.9*dataset_size):]

        self.image_folder = image_folder
        self.text_column1 = text_column1
        self.text_column2 = text_column2
        self.image_id_column1 = image_id_column1
        self.image_id_column2 = image_id_column2
        self.label_name = label_name
        self.image_transform =  transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

    def __len__(self):

        """
        Get the number of samples in the dataset.
        
        Returns:
        - int: Number of samples in the dataset.
        """

        return len(self.dataframe)

    def __getitem__(self, idx):
        """
        Get the data for the sample at the specified index.

        Args:
            idx (int): Index of the current sample.
        
        Returns:
            dict: A dictionary containing the following data for the sample:
                - 'text_data1': Text data from the first column.
                - 'text_data2': Text data from the second column.
                - 'image_data1': Image data corresponding to the first image ID.
                - 'image_data2': Image data corresponding to the second image ID.
                - 'labels': Labels associated with the sample.
        """

        text_data1 = self.dataframe.iloc[idx][self.text_column1]
        text_data2 = self.dataframe.iloc[idx][self.text_column2]

        # Load the corresponding images using ImageFolder
        image_path1 = f"{self.image_folder}/{self.dataframe.iloc[idx][self.image_id_column1]}.jpg"
        image_data1 = Image.open(image_path1).convert("RGB")
        image_path2 = f"{self.image_folder}/{self.dataframe.iloc[idx][self.image_id_column2]}.jpg"
        image_data2 = Image.open(image_path2).convert("RGB")

        labels = self.dataframe.iloc[idx][self.label_name]
        if self.image_transform:
            image_data1 = self.image_transform(image_data1)
            image_data2 = self.image_transform(image_data2)

         # Convert  data to tensors
        labels = torch.tensor(labels)

        return {'text_data1': text_data1, 'text_data2': text_data2, 'image_data1': image_data1, 'image_data2': image_data2, 'labels': labels}
