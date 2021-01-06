import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import UTKFaceDataset
import numpy as np
import torchvision

latent_dim = 50
label_dim = 7

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*512, latent_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        return out

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim+label_dim, 8*8*1024),
            nn.ReLU()
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=1),            
        )

    def forward(self, z, y):
        inp = torch.cat([z, y], dim=1)

        out = self.fc(inp)
        out = out.view(out.shape[0], 1024, 8, 8)
        out = self.net(out)
        return out

class DiscriminatorZ(nn.Module):
    def __init__(self):
        super(DiscriminatorZ, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 1),
        )
    def forward(self, z):
        logit = self.net(z)
        prob = torch.sigmoid(logit)
        return prob

class DiscriminatorX(nn.Module):
    def __init__(self):
        super(DiscriminatorX, self).__init__()

        self.xline = nn.Conv2d(3, 16, kernel_size=2, stride=2)

        self.net = nn.Sequential(
            nn.Conv2d(label_dim+16, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(8*8*128,1024),
            nn.ReLU(),
            nn.Linear(1024,1)
        )
    def forward(self, x, y):
        inp_x = self.xline(x)
        inp_y = torch.cat([y]*64*64, dim = 1)
        inp_y = inp_y.view(y.shape[0], label_dim, 64,64)
        inp_xy = torch.cat([inp_x,inp_y], dim=1)

 
        out = self.net(inp_xy)
        out = out.view(out.shape[0], -1)
        logit = self.fc(out)
        prob = torch.sigmoid(logit)
        
        return prob
    


    


if __name__ == "__main__":

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
    dataset = UTKFaceDataset('./UTKFace', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    encoder = Encoder()
    generator = Generator()
    disz = DiscriminatorZ()
    disx = DiscriminatorX()
    for b, (img, label) in enumerate(dataloader):
        z = encoder(img)

        y = torch.randn(size=(img.shape[0], label_dim))
        
        

        xfake = generator(z, y)
        
        probz = disz(z)
        proby = disx(img, y)
        
        break