import math
from platform import python_branch
from tkinter import NE
from turtle import forward

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

if __name__ == '__main__':
    train_img = np.load('X_train.npy')
    train_label = np.load('y_train.npy')
    test_img = np.load('X_test.npy')
    test_label = np.load('y_test.npy')
    print(train_label.shape)
    print(test_img.shape)
    #dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #print(dev)
    net = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
    #.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    train_img = torch.Tensor(train_img)
    train_label = torch.Tensor(train_label)
    test_img = torch.Tensor(test_img)
    test_label = torch.Tensor(test_label)
    train_dataset = TensorDataset(train_img, train_label)
    test_dataset = TensorDataset(test_img, test_label)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=2,shuffle=False)
    for epoch in range(100):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            optimizer.zero_grad()
            #print(inputs.shape)
            # forward + backward + optimize
            #inputs = inputs.view(3,3,244,244)
            outputs = net(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i %  38707 == 38706:
                print('[{:d}, {:5d}] loss: {:.20f}'
                        .format(epoch + 1, i + 1, running_loss / 38707))
                running_loss = 0.0

    print('Finished Training')
    model_path = 'model.pth'
    torch.save(net.state_dict(), model_path)
    correct = 0
    total = 0

    with torch.no_grad():
        for (images, labels) in test_loader:
            images = images.view(2,3,244,244)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy: {:.2f} %'.format(100 * float(correct/total)))

    