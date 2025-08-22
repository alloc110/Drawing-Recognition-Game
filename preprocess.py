from sklearn.utils import shuffle
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import Mydataset
import matplotlib.pyplot as plt
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from model import ClassificationCNN
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
data = Mydataset('dataset')

data_train = Mydataset('dataset', is_train=True)

len = data_train.__len__()
images = []
labels = []
list = []
for i in data_train:
    image, label = i
    images.append(image)
    labels.append(label)

features = np.array(images)
labels = np.array(labels)
mod = 0.8
features_shuffled, labels_shuffled = shuffle(features, labels, random_state=43)
train = features_shuffled[:int(mod * len)]
train_label =labels_shuffled[:int(mod * len)]
test = features_shuffled[int(mod * len):]
test_label = labels_shuffled[int(mod * len):]

arr_train = np.array(train)
arr_test = np.array(test)
arr_train_label = np.array(train_label)
arr_test_label = np.array(test_label)

np.save("dataset/train.npy", arr_train)
np.save("dataset/test.npy", arr_test)
np.save("dataset/train_label.npy", arr_train_label)
np.save("dataset/test_label.npy", arr_test_label)
print("finish")