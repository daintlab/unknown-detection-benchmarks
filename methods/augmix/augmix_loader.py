import numpy as np

import torch
from torchvision import datasets, transforms

from methods.augmix import augmentations

def aug(image, preprocess, parsers):
    aug_list = augmentations.augmentations

    if parsers['benchmark'] == 'cifar':
        ws = np.float32(np.random.dirichlet([1] * parsers['mixture_width']))
        m = np.float32(np.random.beta(1, 1))
    elif parsers['benchmark'] == 'imagenet':
        ws = np.float32(np.random.dirichlet([parsers['aug_prob_coeff']]
                                            * parsers['mixture_width']))
        m = np.float32(np.random.beta(parsers['aug_prob_coeff'],
                                      parsers['aug_prob_coeff']))

    mix = torch.zeros_like(preprocess(image))
    for i in range(parsers['mixture_width']):
        image_aug = image.copy()
        depth = parsers['mixture_depth'] if parsers['mixture_depth'] > 0 else np.random.randint(1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, parsers['aug_severity'])
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return mixed


class AugMixDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, preprocess, parsers):
        self.dataset = dataset
        self.preprocess = preprocess
        self.parsers = parsers

    def __getitem__(self, i):
        x, y = self.dataset[i]
        im_tuple = (self.preprocess(x), aug(x, self.preprocess, self.parsers),
                    aug(x, self.preprocess, self.parsers))
        return im_tuple, y

    def __len__(self):
        return len(self.dataset)


def loader(root_dir, benchmark, batch_size):
    if benchmark == 'cifar':
        augmentations.IMAGE_SIZE = 32
        dict_augmix = {'batch_size': batch_size,
                       'mixture_width': 3,
                       'mixture_depth': -1,
                       'aug_severity': 3,
                       'all_ops': False,
                       'benchmark': benchmark}

        trn_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4)])
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize([0.507, 0.487, 0.441],
                                  [0.267, 0.256, 0.276])])
        tst_transform = preprocess

        trn_data = datasets.ImageFolder(
            f'{root_dir}/cifar-bench/train/cifar40/',
            transform=trn_transform)
        tst_data = datasets.ImageFolder(
            f'{root_dir}/cifar-bench/test/cifar40/labels/',
            transform=tst_transform)

    elif benchmark == 'imagenet':
        augmentations.IMAGE_SIZE = 224
        dict_augmix = {'batch_size': batch_size,
                       'mixture_width': 3,
                       'mixture_depth': -1,
                       'aug_severity': 1,
                       'all_ops': False,
                       'benchmark': benchmark,
                       'aug_prob_coeff': 1.}

        trn_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip()
        ])
        tst_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
        preprocess = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])

        trn_data = datasets.ImageFolder(
            f'{root_dir}/imagenet-bench/train/imagenet200/',
            transform=trn_transform)
        tst_data = datasets.ImageFolder(
            f'{root_dir}/imagenet-bench/test/imagenet200/labels/',
            transform=tst_transform)

    trn_data = AugMixDataset(trn_data, preprocess, dict_augmix)

    trn_loader = torch.utils.data.DataLoader(
        trn_data,
        batch_size=batch_size,
        shuffle=True)
    tst_loader = torch.utils.data.DataLoader(
        tst_data,
        batch_size=batch_size,
        shuffle=False)

    return trn_loader, tst_loader

