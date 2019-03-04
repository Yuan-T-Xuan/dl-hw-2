import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, models, transforms
from PIL import Image

def vgg19_without_last_layer():
    vgg19 = models.vgg19(pretrained=True)
    new_classifier = torch.nn.Sequential(
        vgg19.classifier[0],
        vgg19.classifier[1],
        vgg19.classifier[3],
        vgg19.classifier[4]
    )
    vgg19.classifier = new_classifier
    vgg19 = vgg19.eval()
    return vgg19

def get_pet_dataloaders():
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    data_dir = '/Users/xuan/Downloads/images'
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'test']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(
            image_datasets[x], batch_size=40, shuffle=True, num_workers=4)
        for x in ['train', 'test']
    }
    return dataloaders
