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

data = Mydataset('dataset')
categories = ['apple',
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
image , label = data.__getitem__(3)
print(type(image))
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)]
)

data = Mydataset('dataset', transforms)

batch_size = 1500

trainloader = DataLoader(data, batch_size=batch_size,
                                      shuffle=False, num_workers=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()
model = ClassificationCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

min_loss = float("inf")
NUM_EPOCHS  = 50
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader,  desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        print(inputs.shape)
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss, epoch)
        if i % 100 ==   99:  # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0
    torch.save(model.state_dict(), 'model/last_model.pt')
    if(running_loss < min_loss):
        min_loss = running_loss
        torch.save(model.state_dict(), 'model/best_model.pt')
# writer.flush()
# writer.close()
#
# print('Finished Training')
