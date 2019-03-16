import numpy as np
import matplotlib.pyplot as plt
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

def image_loader2(pil_image):
    image = pil_image
    image = _image_loader(image).float()
    image = image.unsqueeze(0)
    return image

def image_loader(image_name):
    image = Image.open(image_name)
    image = _image_loader(image).float()
    image = image.unsqueeze(0)
    return image

def get_top_3_2(pil_image):
    image = image_loader2(pil_image)
    result = _model(image)
    result = result.squeeze()
    result = result.tolist()
    i = list(range(len(result)))
    result = list(zip(result, i))
    return sorted(result, reverse=True)[:3]

def get_top_3(image_name):
    image = image_loader(image_name)
    result = _model(image)
    result = result.squeeze()
    result = result.tolist()
    i = list(range(len(result)))
    result = list(zip(result, i))
    return sorted(result, reverse=True)[:3]

def vis_feature_map(layer_idx):
    from collections import OrderedDict
    d = OrderedDict()
    for m in _model.children():
        for n in m.children():
            d[str(len(d))] = n
            if len(d) > layer_idx:
                break
        break
    tmp_model = torch.nn.Sequential(d)
    return tmp_model

def plot_from_tensor(in_tensor, idx):
    img = in_tensor[0][idx].detach().numpy()
    amax = np.amax(img)
    img = img / amax
    plt.imshow(img)
    plt.show()

"""
>>> from prob1 import *
>>> anet = vis_feature_map(1)
>>> in_img = image_loader("peppers.jpg")
>>> out_img = anet(in_img)
>>> out_img.shape
torch.Size([1, 64, 224, 224])
>>> plot_from_tensor(out_img, 32)
>>> for i in range(64):
...     plot_from_tensor(out_img, i)
"""
