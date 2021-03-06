import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision.utils import save_image
from dataloader import UTKFaceDataset
from model import Encoder, DiscriminatorX, DiscriminatorZ,  Generator, latent_dim, label_dim
import numpy as np
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--n_epochs", type=int, default=101, help="Number of epochs")
parse.add_argument("--batch_size", type=int, default=32, help="Size of a batch")
parse.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
parse.add_argument("--image_size", type=int, default=128, help="Size of a batch")
opt = parse.parse_args()


lr = opt.lr 
batch_size = opt.batch_size
n_epochs = opt.n_epochs
image_size = opt.image_size

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(image_size),torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = UTKFaceDataset('./UTKFace', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



encoder = Encoder().to(device)
discriminatorZ = DiscriminatorZ().to(device)
discriminatorX = DiscriminatorX().to(device)
generator = Generator().to(device)

optimizer_EG = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=lr,betas=(0.5, 0.999))
optimizer_Dx = torch.optim.Adam(discriminatorX.parameters(), lr=lr,betas=(0.5, 0.999))
optimizer_Dz = torch.optim.Adam(discriminatorZ.parameters(), lr=lr,betas=(0.5, 0.999))

checkpoint = torch.load('/home/khangt1k25/AgeProgession/checkpoint105.pt', map_location=device)

encoder.load_state_dict(checkpoint['E_state_dict'])
generator.load_state_dict(checkpoint['G_state_dict'])
discriminatorX.load_state_dict(checkpoint['Dx_state_dict'])
discriminatorZ.load_state_dict(checkpoint['Dz_state_dict'])
print("Loaded")




# input: [3, 128, 128]
#
def sampling(image, b, ):
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

        res = torch.randn((10, 3, 128, 128))
        for i in range(label_dim):
            res[i] = generator(zhat, y[:, i,:])[0]
            #res[i] = xhat[0]
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
        save_image(res[i], '/home/khangt1k25/AgeProgession/agegroup%d.png'%i, nrow=5)   


for b, (img, label) in enumerate(dataloader):

    sampling(img[0], b)
    
    if b == 10:
        sampling_group(img)
        break






