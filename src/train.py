from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset import Mydataset
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from model import ClassificationCNN
import torch
from torch.utils.tensorboard import SummaryWriter

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

data_train = Mydataset('data',is_train= True, transforms = transforms)

batch_size = 6000
trainloader = DataLoader(data_train, batch_size=batch_size,
                                      shuffle=True, num_workers=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()

is_continue = True

model = ClassificationCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

if is_continue == True:
    checkpoint = torch.load("../models/last_model.pth5", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']

data_test = Mydataset('data', is_train=False, transforms=transforms)
testloader = DataLoader(data_test, batch_size=batch_size, num_workers=4)
correct = 0
total = 0
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}



min_loss = float("inf")
NUM_EPOCHS  = 200
for epoch in range(start_epoch, NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(tqdm(trainloader,  desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        writer.add_scalar("Loss/train", running_loss, epoch)

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, '../models/last_model.pth5')

    if(running_loss < min_loss):
        min_loss = running_loss
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, '../models/best_model.pth5')
    running_loss = 0.0

    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         # calculate outputs by running images through the network
    #         images = images.to(device)
    #         total += 1
    #         labels = torch.from_numpy(np.array(labels))
    #         labels = labels.to(device)
    #
    #         outputs = models(images)
    #         # the class with the highest energy is what we choose as prediction
    #         loss = criterion(outputs, labels)
    #         _, predicted = torch.max(outputs, 1)
    #         running_loss += loss.item()
    #         writer.add_scalar("Loss/test", running_loss, epoch)
    # running_loss = 0
    # for data in testloader:
    #     images, labels = data
    #     # calculate outputs by running images through the network
    #     images = images.to(device)
    #     total += 1
    #     labels = torch.from_numpy(np.array(labels))
    #     labels = labels.to(device)
    #
    #     outputs = models(images)
    #     # the class with the highest energy is what we choose as prediction
    #     total += labels.size(0)
    #     _, predicted = torch.max(outputs, 1)
    #     for label, prediction in zip(labels, predicted):
    #         if label == prediction:
    #             correct_pred[classes[label]] += 1
    #         total_pred[classes[label]] += 1
writer.flush()
writer.close()

print('Finished Training')
