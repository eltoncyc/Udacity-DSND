from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import argparse


def load_data():
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # TODO: Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    class_labels = train_dataset.classes
    return trainloader, train_dataset, testloader, validloader, class_labels

def load_model():
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(1024, 500)),
                              ('relu1', nn.ReLU()),
                              ('drop1', nn.Dropout(p = 0.2)),
                              ('fc2', nn.Linear(500, 256)),
                              ('relu2', nn.ReLU()),
                              ('drop2', nn.Dropout(p = 0.2)),
                              ('fc3', nn.Linear(256, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    return model

def validation(model, criterion, loader, gpu = False):
    _loss = 0
    _accuracy = 0
    for idx, (images, labels) in enumerate(loader):
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        _, predict = torch.max(outputs.data, 1)
        _loss += criterion(outputs, labels).item()

        _exp = torch.exp(outputs)
        _equality = (labels.data == _exp.max(dim=1)[1])
        _accuracy += _equality.type(torch.FloatTensor).mean()
    return _loss, _accuracy


def train_model(model, learning_rate, epochs, criterion, trainloader, validloader, gpu = False):
    if epochs is None or epochs == 0:
        print("'epochs' should be greater than zero.")
        return

    if gpu:
        model.to('cuda')

    optimizier = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    steps = 0
    print_every = 40

    for e in range(epochs):
        current_loss = 0
        for idx, (inputs, labels) in enumerate(trainloader):
            steps += 1
            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizier.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizier.step()
            current_loss += loss.item()
            if steps % print_every ==0:
                print_loss = current_loss/print_every
                print(f'Epoch: {e+1}/{epochs}, Loss:{print_loss:.5f}')
                current_loss = 0

        model.eval()

        with torch.no_grad():
            valid_loss, valid_accuracy = validation(model = model, criterion=criterion, loader=validloader, gpu = gpu)
            train_loss, train_accuracy = validation(model=model, criterion=criterion, loader=trainloader, gpu = gpu)
            print(f'Epoch:{e+1}/{epochs}',
                 f'Training Loss:{train_loss:.2f}',
                 f'Training Accuracy:{train_accuracy:.2f}',
                 f'Validation Loss:{valid_loss:.2f}',
                 f'Validation Accuracy:{valid_accuracy:.2f}')
        model.train()

def test_model(model, testloader, gpu = False):
    if gpu:
        model.to('cuda')
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            if gpu:
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    network_accuracy = 100 * correct / total
    print(f'Accuracy of the network is {network_accuracy:.2f}%')

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action = 'store_true',
                        dest = 'gpu',
                        default = False,
                        help='GPU if --gpu')

    parser.add_argument('--epochs', action='store',
                        dest = 'epochs',
                        type = int,
                        default = 3,
                        help = 'Number of epochs')

    parser.add_argument('--learning_rate', action = 'store',
                        dest = 'learning_rate',
                        type = float,
                        default = 0.001,
                        help = 'Learning rate')

    settings = parser.parse_args()
    print('=========Training Settings==========')
    print(f'gpu              = {settings.gpu}')
    print(f'epoch(s)         = {settings.epochs}')
    print(f'learning_rate    = {settings.learning_rate}')
    print('====================================')


    trainloader, train_dataset, testloader, validloader, class_labels = load_data()
    model = load_model()
    criterion = nn.NLLLoss()
    optimizier = optim.Adam(model.classifier.parameters(), lr = settings.learning_rate)
    train_model(model, settings.learning_rate, settings.epochs, criterion, trainloader, validloader, gpu = settings.gpu)
    test_model(model, testloader, gpu = settings.gpu)
    model.to('cpu')

    model.class_to_idx = train_dataset.class_to_idx
    checkpoint = {'classifier': model.classifier,
             'class_to_idx': model.class_to_idx,
             'state_dict': model.state_dict(),
             'optimizer': optimizier.state_dict(),
             'class_labels': class_labels}
    save_checkpoint(checkpoint)

if __name__ == '__main__':
    main()
