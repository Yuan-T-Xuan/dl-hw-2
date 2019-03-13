import torch
from torch.autograd import Variable
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from prob1 import image_loader

in_image = "peppers.jpg"
target_category = 812
model = models.vgg19(pretrained=True)
lmbda = 0.1
epsilon = 0.2

def get_top_3(image, model):
    result = model(image)
    result = result.squeeze()
    result = result.tolist()
    i = list(range(len(result)))
    result = list(zip(result, i))
    return sorted(result, reverse=True)[:3]

def show_image(x, rescale=False):
    x = x[0]
    x = np.array(x.tolist())
    x[0] = x[0] * 0.229 + 0.485
    x[1] = x[1] * 0.224 + 0.456
    x[2] = x[2] * 0.225 + 0.406
    if rescale:
        for i in range(3):
            x[i] = x[i] - np.amin(x[i])
            x[i] = x[i] / np.amax(x[i])
    plt.imshow(x.transpose(1,2,0))
    plt.show()

x = image_loader(in_image)
show_image(x)
x = Variable(x, requires_grad=True)

original_image = image_loader(in_image)

for i in range(60):
    print("i: " + str(i))
    res = model(x)
    res[0, target_category].backward()
    x = x + lmbda * x.grad
    delta = x - original_image
    delta = torch.clamp(delta, -epsilon, epsilon)
    x = original_image + delta
    x[0][0].clamp_(-2.1179039301310043, 2.2489082969432315)
    x[0][1].clamp_(-2.0357142857142856, 2.428571428571429)
    x[0][2].clamp_(-1.8044444444444445, 2.6399999999999997)
    x = Variable(x, requires_grad=True)

print(get_top_3(x, model))
show_image(x, True)

plt.imshow(np.array((x - original_image)[0].tolist()).transpose(1,2,0))
plt.show()
