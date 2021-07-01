import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataloader
import torchvision
from torchvision.utils import save_image
from datasets import UTK_dataset
from model import Encoder, DiscriminatorX, DiscriminatorZ,  Generator, latent_dim, label_dim
import numpy as np
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("--n_epochs", type=int, default=101, help="Number of epochs")
parse.add_argument("--batch_size", type=int, default=32, help="Size of a batch")
parse.add_argument("--lr", type=float, default=0.0002, help="Learning rate")
parse.add_argument("--image_size", type=int, default=128, help="Size of an image")
opt = parse.parse_args()



lr = opt.lr 
batch_size = opt.batch_size
n_epochs = opt.n_epochs
image_size = opt.image_size


dataset = UTK_dataset('./compress/train_utk.pkl',)
dataloader = Dataloader(dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = Encoder().to(device)
discriminatorZ = DiscriminatorZ().to(device)
discriminatorX = DiscriminatorX().to(device)
generator = Generator().to(device)

optimizer_EG = torch.optim.Adam(list(encoder.parameters())+list(generator.parameters()), lr=lr,betas=(0.5, 0.999))
optimizer_Dx = torch.optim.Adam(discriminatorX.parameters(), lr=lr,betas=(0.5, 0.999))
optimizer_Dz = torch.optim.Adam(discriminatorZ.parameters(), lr=lr,betas=(0.5, 0.999))

mse_loss = nn.MSELoss()
adv_loss = nn.BCELoss()
l1_loss = nn.L1Loss()



for epoch in range(n_epochs):
    EG = 0
    X = 0
    Z = 0
    
    for batch, (image, label) in enumerate(dataloader):

        valid = Variable(torch.Tensor(image.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(torch.Tensor(image.shape[0], 1).fill_(0.0), requires_grad=False).to(device)
        
        xreal = Variable(torch.Tensor(image)).to(device)

        one_hot = torch.nn.functional.one_hot(label, num_classes=label_dim).type(torch.float32)
        one_hot[one_hot==0] = -1
        y = Variable(one_hot).to(device)
        z = Variable(torch.Tensor(np.random.uniform(-1., 1., size=(image.shape[0], latent_dim)))).to(device)

        '''
        optimizing E G
        '''
        optimizer_EG.zero_grad()
            
        zhat = encoder(xreal)
        xhat = generator(zhat, y)
        
        loss_EG =   0.0001 * adv_loss(discriminatorX(xhat, y), valid) +\
                    0.0001 *adv_loss(discriminatorZ(zhat), valid) + 0.01*total_variation_loss(xhat)+\
                    mse_loss(xreal, xhat)
                    # 0.1*l1_loss(xreal, xhat) + 0.9*mse_loss(xreal, xhat)
    
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


        EG += loss_EG.item()
        Z += loss_Dz.item()
        X += loss_Dx.item()

    print("Epoch: {}|  Loss EG: {} | lossZ: {}| lossX:  {}".format(epoch,EG/(batch+1), Z/(batch+1), X/(batch+1)))

    
    
    if epoch % 20 ==0 and epoch != 0:
        save_image(xhat[:25], './gen%d.png'%epoch, nrow=5)
        save_image(xreal[:25], './real%d.png'%epoch, nrow=5)

    if epoch % 50 == 0 and epoch != 0:
  
        path = "./checkpoint"+str(epoch)+".pt"
        checkpoint = torch.save({
            'epoch': epoch,
            'G_state_dict': generator.state_dict(),
            'Dx_state_dict': discriminatorX.state_dict(),
            'E_state_dict': encoder.state_dict(),
            'Dz_state_dict': discriminatorZ.state_dict(),
            'optimizerDx_state_dict': optimizer_Dx.state_dict(),
            'optimizerDz_state_dict':optimizer_Dz.state_dict(),
            'optimizerEG_state_dict': optimizer_EG.state_dict()
        }, path)
        print("Saving...")
            







