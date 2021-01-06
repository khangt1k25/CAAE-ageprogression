import torch
import torchvision
import os
import glob
from torchvision import transforms


class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = self._prepare_samples(root)
    def __getitem__(self, index):
        path, label = self.samples[index]
        image = torchvision.datasets.folder.pil_loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.samples)
    
    def _prepare_samples(self, root):
        samples = []
        os.chdir(root)
        paths  = glob.glob("*.jpg")
        for path in paths[:1000]:
            try:
                label = self._load_label_(path)
            except Exception as e:
                print('path: {}, exception: {}'.format(path, e))
                continue

            samples.append((path, label))
        return samples
    def _load_label_(self, path):
        str_list = os.path.basename(path).split('.')[0].strip().split('_')
        age = int(str_list[0])
        if age < 10:
            label = 0
        elif age < 20:
            label = 1
        elif age < 30:
            label = 2
        elif age < 40:
            label = 3
        elif age < 50:
            label = 4
        elif age < 60:
            label = 5
        else:
            label = 6
        return label

if __name__ == "__main__":

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize(128)])
    dataset = UTKFaceDataset('./UTKFace', transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    

    for b, (img, label) in enumerate(dataloader):
        print(b)
        print(img)
        print(img.shape)
        print(label)
        break