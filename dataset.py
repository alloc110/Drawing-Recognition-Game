import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class Mydataset(Dataset):
    def __init__(self, path = 'dataset', is_train = True , transforms = None   ):
        self.list_file = os.listdir(path)
        labels = ['apple',
                      'cake',
                      'rabbit',
                      'fish',
                      'bread',
                      'monkey',
                      'hat',
                      'elephant',
                      'bird',
                      'lion',
                      'car'
                      ]
        self.images = []
        self.labels = []
        if is_train == True:
            data = np.load(path + "/train.npy")
            for d in data:
                self.images.append(d.reshape(28, 28))
            labels = np.load(path + '/train_label.npy')
            for i in labels:
                self.labels.append(i)
        else:
            data = np.load(path + "/test.npy")
            for d in data:
                self.images.append(d.reshape(28, 28))
            labels = np.load(path + '/test_label.npy')
            for i in labels:
                self.labels.append(i)

        self.transforms = transforms


    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image[image > 180] = 255
        image[image <= 180] = 0
        if self.transforms:
            image = self.transforms(image)
        return image, label

    def __len__(self):
        return len(self.labels)
