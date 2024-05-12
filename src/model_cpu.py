# Code taken from pytorch tutorial on image classification 

## Load and normalize the CIFAR10 training and test datasets using torchvision
## Define a Convolutional Neural Network
## Define a loss function
## Train the network on the training data
## Test the network on the test data

import numpy as np
import torch 
import torchvision as trv
import torchvision.transforms as trf
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import NeuralNet
import matplotlib.pyplot as plt

### Define a convulutional neural network class
    
def main():
    ### Import test image dataset
    transform = trf.Compose([trf.ToTensor(), trf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4
    trainset = trv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = trv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    MODEL_PATH = './model_pytorch/output/'
    
    train_model(trainloader, MODEL_PATH)
    test_model(testloader, MODEL_PATH)

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
   
def test_model(testloader, path):

    net = NeuralNet.CNN()
    net.load_state_dict(torch.load(str(path+r'.\cnn_model.pth')))

    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(4)))
    
    #imshow(trv.utils.make_grid(images))

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


def train_model(trainloader, path):
    net = NeuralNet.CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
    torch.save(net.state_dict(), str(path+r'.\cnn_model.pth'))
    print('Finished Training')


if __name__ == '__main__':
    main()
