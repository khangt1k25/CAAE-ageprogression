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

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=51, help="number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="size of batch")
parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
parser.add_argument("--latent_dim", type=int, default=50, help="dim of latent variable z")
parser.add_argument("--label_dim", type=int, default=7, help="dim of label one-hot, num classes of age")
parser.add_argument("--image_shape", type=tuple, default=(3, 128, 128), help="size of an image (c, h, w)")
parser.add_argument("--lamda", type=int, default=100, help="Lamda  for reconstruction loss")
parser.add_argument("--chita", type=int, default=10, help="Chita for total varition loss")

opt = parser.parse_args()





transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
dataset = UTKFaceDataset('./UTKFace', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

encoder = Encoder().to(device)
discriminatorZ = DiscriminatorZ().to(device)
discriminatorX = DiscriminatorX().to(device)
generator = Generator().to(device)

optimizer_EG = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=opt.lr)
# optimizer_E = torch.optim.Adam(encoder.parameters(), lr=opt.lr)
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr)
optimizer_Dx = torch.optim.Adam(discriminatorX.parameters(), lr=opt.lr)
optimizer_Dz = torch.optim.Adam(discriminatorZ.parameters(), lr=opt.lr)

mse_loss = nn.MSELoss().to(device)
adv_loss = nn.BCELoss().to(device)

def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def sampling(img):
    label = range(0, label_dim)
    label = np.array([label]*25)
    label = torch.as_tensor(label)
    y = Variable(torch.nn.functional.one_hot(label, num_classes=opt.label_dim).type(torch.float32)).to(device)
    xreal = Variable(torch.Tensor(image[:25])).to(device)
    zhat = encoder(xreal)

    for i in range(label_dim):
        xhat = generator(zhat, y[:,i,:])
        save_image(xhat,'agegroup%d.png'%i, nrow=5)

def train():
    for epoch in range(opt.n_epochs):
        for batch, (image, label) in enumerate(dataloader):
            
            valid = Variable(torch.Tensor(image.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
            fake = Variable(torch.Tensor(image.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
            
            xreal = Variable(torch.Tensor(image)).to(device)
            y = Variable(torch.nn.functional.one_hot(label, num_classes=opt.label_dim).type(torch.float32)).to(device)
            z = Variable(torch.Tensor(np.random.uniform(size=(image.shape[0],opt.latent_dim)))).to(device)

            '''
            optimizing EG
            ''' 
            optimizer_EG.zero_grad()
            
            zhat = encoder(xreal)
            xhat = generator(zhat, y)
            
            loss_EG = opt.lamda * mse_loss(xreal, xhat) + adv_loss(discriminatorX(xhat, y), valid) +\
                        adv_loss(discriminatorZ(zhat), valid) + opt.chita *total_variation_loss(xhat)
        
            loss_EG.backward()
            optimizer_EG.step()
            

            '''
            optimizing Dz
            '''
            optimizer_Dz.zero_grad()
            loss_Dz = adv_loss(discriminatorZ(z), valid) + adv_loss(discriminatorZ(zhat.detach()), fake)
            loss_Dz.backward()
            optimizer_Dz.step()

            '''
            optimizing Dx
            '''
            optimizer_Dx.zero_grad()
            loss_Dx = adv_loss(discriminatorX(xreal, y), valid) + adv_loss(discriminatorX(xhat.detach(), y), fake)
            loss_Dx.backward()
            optimizer_Dx.step()
        
        EG = loss_EG.item()
        Z = loss_Dz.item()
        X = loss_Dx.item()

        print("Loss EG {} | loss Z {}| loss X  {}".format(EG, Z, X))
        if epoch%10==0:
            save_image(xhat[:25], './epoch%d.png'%epoch, nrow=5)
            

if __name__ == "__main__":
    for batch, (image, label) in enumerate(dataloader):
        sampling(image[:25])
        break

