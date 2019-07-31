from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from collections import OrderedDict
import torch
from torch import nn
from torch import optim
import argparse
from PIL import Image
import json


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

def load_checkpoint(filename = 'checkpoint.pth.tar'):
    checkpoint = torch.load(filename)
    model = load_model()
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = optim.Adam(model.classifier.parameters())
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, checkpoint['class_labels']

def load_json(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img_loader = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    pil_image = Image.open(image)
    pil_image = img_loader(pil_image).float()
    np_image = np.array(pil_image)
    return np_image

def predict(image_path, model, topk, class_labels, cat_to_name, gpu = False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    image_tensor.resize_([1, 3, 224, 224])
    model.to('cpu')

    if gpu is True:
        model.to('cuda')
        image_tensor = image_tensor.to('cuda')

    result = torch.exp(model(image_tensor))
    ps, index = result.topk(topk)
    ps, index = ps.detach(), index.detach()
    ps.resize_([topk])
    index.resize_([topk])
    ps, index = ps.tolist(), index.tolist()

    label_index = []
    for i in index:
        label_index.append(int(class_labels[int(i)]))

    labels = []
    for i in label_index:
        labels.append(cat_to_name[str(i)])

    return ps, labels, label_index


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action = 'store_true',
                        dest = 'gpu',
                        default = False,
                        help='GPU if --gpu')

    parser.add_argument('--topk', action='store',
                        dest = 'topk',
                        type = int,
                        default = 5,
                        help = 'Top K possibilities')

    parser.add_argument('--img', action = 'store',
                        dest = 'img',
                        type = str,
                        default = 'sample_img_11.jpg',
                        help = 'Store Img Name')

    parser.add_argument('--category_names', action = 'store',
                        dest = 'category',
                        type = str,
                        default = 'cat_to_name.json',
                        help = 'name to map categories')


    settings = parser.parse_args()
    print('=========Prediction Settings==========')
    print(f'gpu             = {settings.gpu}')
    print(f'img             = {settings.img}')
    print(f'topk            = {settings.topk}')
    print(f'category        = {settings.category}')
    print('======================================')

    model, class_labels = load_checkpoint()
    cat_to_name = load_json(settings.category)
    ps, labels, index = predict(settings.img, model, settings.topk, class_labels, cat_to_name,  settings.gpu)
    print("===================Predictions===================")
    for i in range(len(ps)):
        print(f"The probability of the class to be {labels[i]} is {ps[i] * 100:.2f} %.")

if __name__ == '__main__':
    main()
