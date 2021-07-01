import torch
import torch.nn
def total_variation_loss(imgTensor, image_size):

    x = (imgTensor[:,:,1:,:]-imgTensor[:,:,:image_size-1,:])**2
    y = (imgTensor[:,:,:,1:]-imgTensor[:,:,:,:image_size-1])**2

    out = (x.mean(dim=2)+y.mean(dim=3)).mean()
    return out


