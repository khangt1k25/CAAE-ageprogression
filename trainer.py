import torch
import torch.nn as nn
from torch.autograd import Variable
#from torchvision.utils import save_image
import numpy as np
from model import label_dim, latent_dim
from utils import total_variation_loss

mse_loss = nn.MSELoss()
adv_loss = nn.BCELoss()
l1_loss = nn.L1Loss()




class Trainer():
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.encoder, self.generator, self.discriminatorX, self.discriminatorZ = self.model
        self.optimizer_EG, self.optimizer_Dx, self.optimizer_Dz = self.optimizer
        self.device = device

    def train(self, trainloader, e_start, e_end):
        for epoch in range(e_start, e_end+1, 1):
            EG = 0
            X = 0
            Z = 0
            
            for batch, (image, label) in enumerate(trainloader):

                valid = Variable(torch.Tensor(image.shape[0], 1).fill_(1.0), requires_grad=False).to(self.device)
                fake = Variable(torch.Tensor(image.shape[0], 1).fill_(0.0), requires_grad=False).to(self.device)
                
                xreal = Variable(torch.Tensor(image)).to(self.device)

                one_hot = torch.nn.functional.one_hot(label, num_classes=label_dim).type(torch.float32)
                one_hot[one_hot==0] = -1
                y = Variable(one_hot).to(self.device)
                z = Variable(torch.Tensor(np.random.uniform(-1., 1., size=(image.shape[0], latent_dim)))).to(self.device)

                '''
                optimizing E G
                '''
                self.optimizer_EG.zero_grad()
                    
                zhat = self.encoder(xreal)
                xhat = self.generator(zhat, y)
                
                loss_EG =   0.0001 * adv_loss(self.discriminatorX(xhat, y), valid) +\
                            0.0001 *adv_loss(self.discriminatorZ(zhat), valid) + 0.01*total_variation_loss(xhat)+\
                            mse_loss(xreal, xhat)
                            # 0.1*l1_loss(xreal, xhat) + 0.9*mse_loss(xreal, xhat)
            
                loss_EG.backward()
                self.optimizer_EG.step()
                
                '''
                optimizing Dz
                '''
                self.optimizer_Dz.zero_grad()
                loss_Dz = adv_loss(self.discriminatorZ(z), valid) + adv_loss(self.discriminatorZ(zhat.detach()), fake)
                loss_Dz.backward()
                self.optimizer_Dz.step()

                '''
                optimizing Dx
                '''
                self.optimizer_Dx.zero_grad()
                loss_Dx = adv_loss(self.discriminatorX(xreal, y), valid) + adv_loss(self.discriminatorX(xhat.detach(), y), fake)
                loss_Dx.backward()
                self.optimizer_Dx.step()


                EG += loss_EG.item()
                Z += loss_Dz.item()
                X += loss_Dx.item()

            print("Epoch: {} |  Loss EG: {} | lossZ: {} | lossX:  {}".format(epoch,EG/(batch+1), Z/(batch+1), X/(batch+1)))

            
            
            # if epoch % 20 ==0 and epoch != 0:
            #     save_image(xhat[:25], './gen%d.png'%epoch, nrow=5)
            #     save_image(xreal[:25], './real%d.png'%epoch, nrow=5)

            if epoch % 10 == 0:
                self.saving(epoch)
                
                
    def saving(self, epoch):
        try:
            path = "./checkpoint/model"+str(epoch)+".pt"
            checkpoint = torch.save({
                'epoch': epoch,
                'G_state_dict': self.generator.state_dict(),
                'Dx_state_dict': self.discriminatorX.state_dict(),
                'E_state_dict': self.encoder.state_dict(),
                'Dz_state_dict': self.discriminatorZ.state_dict(),
                'optimizerDx_state_dict': self.optimizer_Dx.state_dict(),
                'optimizerDz_state_dict': self.optimizer_Dz.state_dict(),
                'optimizerEG_state_dict': self.optimizer_EG.state_dict()
            }, path)
            print("Save successful")
        except:
            print("Fail to save")

    def load(self, epoch):
        try:
            path = "./checkpoint/model"+str(epoch)+".pt"
            checkpoint = torch.load(path)
      
            self.generator.load_state_dict(checkpoint['G_state_dict'])
            self.encoder.load_state_dict(checkpoint['E_state_dict'])
            self.discriminatorX.load_state_dict(checkpoint['Dx_state_dict'])
            self.discriminatorZ.load_state_dict(checkpoint['Dz_state_dict'])
            self.optimizer_Dx.load_state_dict(checkpoint['optimizerDx_state_dict'])
            self.optimizer_Dz.load_state_dict(checkpoint['optimizerDz_state_dict'])
            self.optimizer_EG.load_state_dict(checkpoint['optimizerEG_state_dict'])

            print("Load successful at epoch " +str(epoch))
        except:
            print("Fail to load")







