import os

import torch
from torchvision import datasets, transforms

def trn_out_loader(root_dir, benchmark, batch_size):
    if benchmark == 'cifar':
        out_dir = os.path.join(root_dir, 'cifar-bench/train/new-tinyimagenet158/no-labels/')

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                         std=[0.267, 0.256, 0.276])

        oe_dataset = datasets.ImageFolder(out_dir,
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))

    elif benchmark == 'imagenet':
        out_dir = os.path.join(root_dir, 'imagenet-bench/train/external-imagenet394/no-labels/')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        oe_dataset = datasets.ImageFolder(out_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader_out = torch.utils.data.DataLoader(oe_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)

    return train_loader_out


