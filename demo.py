import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
from dataloader import loading_dataloader
from model import Encoder, DiscriminatorX, DiscriminatorZ,  Generator, latent_dim, label_dim
import numpy as np
import argparse

from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder().to(device)
discriminatorZ = DiscriminatorZ().to(device)
discriminatorX = DiscriminatorX().to(device)
generator = Generator().to(device)


checkpoint = torch.load('/home/khangt1k25/AgeProgession/checkpoint.pt', map_location=device)

encoder.load_state_dict(checkpoint['E_state_dict'])
generator.load_state_dict(checkpoint['G_state_dict'])
discriminatorX.load_state_dict(checkpoint['Dx_state_dict'])
discriminatorZ.load_state_dict(checkpoint['Dz_state_dict'])
print("Loaded")




# input: [3, 128, 128]
def sampling(image, b ):
    with torch.no_grad():

        print('Sampling')
        an_img = torch.unsqueeze(image, 0)

        label = range(0, label_dim)

        label = np.array([label])

        label = torch.as_tensor(label)

        one_hot = torch.nn.functional.one_hot(label, num_classes=label_dim).type(torch.float32)

        one_hot[one_hot==0]  = -1.


        y = Variable(one_hot).to(device)
        xreal = Variable(torch.Tensor(an_img)).to(device)
        zhat = encoder(xreal)

        res = torch.randn((label_dim, 3, 128, 128))
        for i in range(label_dim):
            res[i] = generator(zhat, y[:, i,:])[0]
        
        save_image(res,'/home/khangt1k25/AgeProgession/testimage%d.png'%b, nrow=1)




def sampling_group(img):
    label = range(0, label_dim)
    label = np.array([label]*25)
    label = torch.as_tensor(label)
    one_hot = torch.nn.functional.one_hot(label, num_classes=label_dim).type(torch.float32)
    one_hot[one_hot==0] = -1.
    y = Variable(one_hot).to(device)
    xreal = Variable(torch.Tensor(img[:25])).to(device)
    zhat = encoder(xreal)

    
    res = []
    for i in range(label_dim):
        xhat = generator(zhat, y[:,i,:])
        res.append(xhat)

    for i in range(label_dim):
        save_image(res[i], '/home/khangt1k25/AgeProgession/agegroup%d.jpg'%i, nrow=5)   



transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((128,128)),torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
img_tensor = torchvision.datasets.folder.pil_loader('/home/khangt1k25/AgeProgession/test3.jpg')
print(img_tensor)
print()
img_tensor  = transform(img_tensor)
print(img_tensor)
print(type(img_tensor))
print(img_tensor.shape)
print('labeldim', label_dim)
save_image(img_tensor, '/home/khangt1k25/AgeProgession/origin.jpg', nrow=1)
sampling(img_tensor, 11)
