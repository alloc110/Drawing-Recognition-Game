import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Mydataset(Dataset):
    def __init__(self, path = 'dataset', transforms = None   ):
        self.list_file = os.listdir(path)
        labels = ['apple',
                      'bird',
                      'bread',
                      'cake',
                      'car',
                      'elephant',
                      'fish',
                      'hat',
                      'lion',
                      'monkey',
                      'rabbit'
                      ]
        self.images = []
        self.labels = []
        self.transforms = transforms
        for i, file in enumerate(self.list_file):
            data = np.load(path + "/" + file)
            for d in data:
                self.images.append(d.reshape(28,28))
                self.labels.append(i)

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)
