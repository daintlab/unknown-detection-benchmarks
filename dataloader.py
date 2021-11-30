import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def trn_loader(root_dir, benchmark, batch_size):

    if benchmark == 'cifar':
        data_dir = os.path.join(root_dir, 'cifar-bench/train/cifar40/')

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        trn_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])

    elif benchmark == 'imagenet':
        data_dir = os.path.join(root_dir, 'imagenet-bench/train/imagenet200/')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        trn_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)
        ])


    trn_dataset = datasets.ImageFolder(data_dir, transform=trn_transforms)
    trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=4)

    return trn_loader


def tst_loader(root_dir, benchmark, batch_size, mode):

    if benchmark == 'cifar':
        data_dir = os.path.join(root_dir,
                                f'cifar-bench/{mode}/cifar40/labels/')

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        tst_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean,
                                                                 std=stdv)])

    elif benchmark == 'imagenet':
        data_dir = os.path.join(root_dir,
                                f'imagenet-bench/{mode}/imagenet200/labels/')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        tst_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)])

    tst_dataset = datasets.ImageFolder(data_dir,
                                       transform=tst_transform)

    tst_loader = torch.utils.data.DataLoader(tst_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)

    return tst_loader



def in_dist_loader(root_dir, data, batch_size, mode):

    if data == 'cifar':
        data_dir = os.path.join(root_dir, f'cifar-bench/{mode}/cifar40/no-labels/')

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        test_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean,
                                                                  std=stdv)])

    elif data == 'imagenet':
        data_dir = os.path.join(root_dir, f'imagenet-bench/{mode}/imagenet200/no-labels/')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)])

    in_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    in_loader = torch.utils.data.DataLoader(in_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4)

    return in_loader


def out_dist_loader(root_dir, benchmark, data, batch_size, mode):

    if benchmark == 'cifar':

        mean = [0.507, 0.487, 0.441]
        stdv = [0.267, 0.256, 0.276]

        data_dir = os.path.join(root_dir, f'cifar-bench/{mode}/{data}/no-labels/')

        if data in ['new-tinyimagenet158', 'describable-textures']:
            test_transform = transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=mean,
                                                     std=stdv)])

        else:
            test_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=mean,
                                                     std=stdv)])

    elif benchmark == 'imagenet':
        data_dir = os.path.join(root_dir, f'imagenet-bench/{mode}/{data}/no-labels/')

        mean = [0.485, 0.456, 0.406]
        stdv = [0.229, 0.224, 0.225]

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv)])

    out_dataset = datasets.ImageFolder(data_dir, transform=test_transform)
    out_loader = torch.utils.data.DataLoader(out_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4)
    return out_loader