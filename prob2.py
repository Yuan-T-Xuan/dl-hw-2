import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import optim, nn
from torchvision import datasets, models, transforms
from PIL import Image

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc1 = nn.Linear(4096, 37)
    
    def forward(self, x):
        return self.fc1(x)

NUM_EPOCHS = 5
linearnet = LinearNet().cuda()
criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(linearnet.parameters(), lr=0.01, momentum=0.8)

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
            image_datasets[x], batch_size=40, shuffle=True, num_workers=8)
        for x in ['train', 'test']
    }
    return dataloaders

dloader = get_pet_dataloaders()
trainloader = dloader['train']
testloader = dloader['test']
vgg = vgg19_without_last_layer().cuda()

for epoch in range(NUM_EPOCHS):
    for i, data in enumerate(trainloader, 0):
        #print("data loaded")
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs = vgg(inputs)
        #print("vgg passed")
        q = torch.norm(inputs, dim=1)
        q = q.reshape((len(q), 1)).detach()
        inputs = inputs / q
        optimizer.zero_grad()
        #print("normalized")
        outputs = linearnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #print("one mini-batch finished")
        #
        print('[%5d, %5d] loss: %.3f' % (epoch + 1, i + 1, loss.item()))