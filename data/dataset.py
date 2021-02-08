"""
Author: Seph Pace
Email:  sephpace@gmail.com
"""

import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from settings import ROOT_DIR


class Fruits360Dataset(Dataset):
    """
    The Fruits 360 dataset.

    Attributes:
        data (list of tuple): The paths to each image in the dataset and its
                              associated label.
        labels (list of str): Labels of fruit types.
    """

    def __init__(self, test=False):
        """
        Constructor.

        Parameters:
            test (bool): Will use the testing dataset if True and the training
                         dataset if False.
        """
        # Distinguish between testing and training data
        if(test):
            data_dir = f'{ROOT_DIR}/data/fruits-360/Testing'
        else:
            data_dir = f'{ROOT_DIR}/data/fruits-360/Training'

        # Save the labels
        self.labels = sorted(os.listdir(data_dir))

        # Find all file paths for each image
        self.data = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                path = f'{root}/{file}'
                if os.path.isfile(path):
                    label = path.split('/')[-2]
                    target = self.labels.index(label)
                    self.data.append((path, target))

    def __getitem__(self, idx):
        """
        Get the image at the specified index in the dataset.

        Parameters:
            idx (int): The index.

        Returns:
            (Tensor): An image tensor.
        """
        path, label = self.data[idx]
        img = Image.open(path)
        tensor = torch.as_tensor(np.asarray(img)).float() / 255
        tensor = tensor.transpose(1, 2).transpose(0, 1)
        return tensor, label
    
    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            (int): The length of the dataset.
        """
        return len(self.data) 
