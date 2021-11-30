import os
from PIL import Image

import torch
from torchvision import datasets, transforms


class idxDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, transform=None):
        self.x_data = x
        self.y_data = y
        self.transform = transform

    def __len__(self):
        return len(self.x_data)

    # return idx
    def __getitem__(self, idx):
        img = Image.open(self.x_data[idx][0])
        img = img.convert('RGB')
        x = self.transform(img)

        return x, self.y_data[idx], idx



def loader(root_dir, benchmark, batch_size):
    if benchmark == 'cifar':
        trn_dir = os.path.join(root_dir, 'cifar-bench/train/cifar40/')
        tst_dir = os.path.join(root_dir, 'cifar-bench/test/cifar40/labels/')

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        trn_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])

        tst_transforms = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=stdv)])


    elif benchmark == 'imagenet':
        trn_dir = os.path.join(root_dir, 'imagenet-bench/train/imagenet200/')
        tst_dir = os.path.join(root_dir, 'imagenet-bench/test/imagenet200/labels/')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        trn_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
            ])

        tst_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)])


    trn_dataset = datasets.ImageFolder(trn_dir,
                                       transform=trn_transforms)
    trn_dataset = idxDataset(trn_dataset.samples,
                             trn_dataset.targets,
                             transform=trn_transforms)

    tst_dataset = datasets.ImageFolder(tst_dir,
                                       transform=tst_transforms)
    tst_dataset = idxDataset(tst_dataset.samples,
                             tst_dataset.targets,
                             transform=tst_transforms)


    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)

    tst_loader = torch.utils.data.DataLoader(tst_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    return trn_loader, tst_loader