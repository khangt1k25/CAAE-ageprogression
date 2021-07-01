import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataloader
from torchvision.utils import save_image
from datasets import UTK_dataset
from model import Encoder, DiscriminatorX, DiscriminatorZ,  Generator
from trainer import Trainer

import numpy as np
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--n_epochs", type=int, default=101, help="Number of epochs")
parse.add_argument("--batch_size", type=int, default=32, help="Size of a batch")
parse.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
opt = parse.parse_args()
lr = opt.lr 
batch_size = opt.batch_size
n_epochs = opt.n_epochs

if __name__ == '__main__':
    


    dataset = UTK_dataset('./compress/train_utk.pkl')
    dataloader = Dataloader(dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder().to(device)
    discriminatorZ = DiscriminatorZ().to(device)
    discriminatorX = DiscriminatorX().to(device)
    generator = Generator().to(device)

    optimizer_EG = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=lr,betas=(0.5, 0.999))
    optimizer_Dx = torch.optim.Adam(discriminatorX.parameters(), lr=lr,betas=(0.5, 0.999))
    optimizer_Dz = torch.optim.Adam(discriminatorZ.parameters(), lr=lr,betas=(0.5, 0.999))


    model = (encoder, generator, discriminatorX, discriminatorZ)
    optimizer = (optimizer_EG, optimizer_Dx, optimizer_Dz)

    trainer = Trainer(model, optimizer, device)
    trainer.train(dataloader, 1, 100)








