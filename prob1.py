import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

imsize = 224
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
_image_loader = transforms.Compose([
    transforms.Resize((imsize, imsize)), 
    transforms.ToTensor(), 
    normalize
])
_model = models.vgg19(pretrained=True)

def image_loader(image_name):
    image = Image.open(image_name)
    image = _image_loader(image).float()
    image = image.unsqueeze(0)
    return image

def get_top_3(image_name):
    image = image_loader(image_name)
    result = _model(image)
    result = result.squeeze()
    result = result.tolist()
    i = list(range(len(result)))
    result = list(zip(result, i))
    return sorted(result, reverse=True)[:3]

