import torch
from torch.utils import data
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import pickle

image_size = 128


# def loading_dataloader(des_dir = "./data/", image_size=128, batch_size = 32):



#     dataset = torchvision.datasets.ImageFolder(root=des_dir,
#                                transform=transforms.Compose([
#                                    transforms.Resize(image_size),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                ]))

#     dataloader = torch.utils.data.DataLoader(dataset,
#                                              batch_size= batch_size,
#                                              shuffle=True)

#     return dataloader


    

UTK_transforms = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  
])


class UTK_dataset(Dataset):
    def __init__(self, path_to_compress_file, transform=None):

        with open(path_to_compress_file, 'rb') as f:
            self.files = pickle.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label, img = self.files[idx]['label'], self.files[idx]['img']

        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        
        return img, label


    