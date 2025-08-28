import numpy as np
import torch
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from src.dataset import Mydataset
from model import ClassificationCNN

path_model = 'models/last_model.pt'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClassificationCNN()
checkpoint = torch.load('../models/best_model.pth5')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)]
)
classes = ['apple',
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

data_test = Mydataset('data', is_train=False, transforms=transforms)
dataloader = DataLoader(data_test, batch_size=5000, num_workers=8)
correct = 0
total = 0
table = [[0 for _ in range(11)] for _ in range(11)]
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in dataloader:
        images, labels = data
        # calculate outputs by running images through the network
        images = images.to(device)
        total += 1
        labels = torch.from_numpy(np.array(labels))
        labels = labels.to(device)

        outputs = model(images)
        # the class with the highest energy is what we choose as prediction
        total += labels.size(0)
        _, predicted = torch.max(outputs, 1)
        for label, prediction in zip(labels, predicted):
            table[label][prediction] += 1


# image, label = data_test.__getitem__(4)
# plt.imshow(image)
# print(classes[label])
# plt.show()

sns.heatmap(table)
plt.show()