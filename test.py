import os
import torch
import torchvision


filename = os.listdir('./UTKFace')


#filename = [name.split('_') for name in filename]
age = [int(name.split('_')[0]) for name in filename]
image_dict = {}

for i, name in enumerate(filename):
    image_dict[name] = age[i]


print(image_dict)