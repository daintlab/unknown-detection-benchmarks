import os
import argparse
import utils

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from metrics_md import md_metrics
from metrics_ood import ood_metrics


def main(args):
    cudnn.benchmark = True

    # Check whether cuda is available
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Create save directory
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Set parameters according to the benchmark choice
    if args.benchmark == 'cifar':
        from models.resnet_cifar import resnet20
        model_fn = resnet20
        num_classes = 40
        last_epoch = 200
        out_data = {'VAL': 'new-tinyimagenet158',
                    'NEAR': 'cifar60',
                    'OOD1': 'svhn',
                    'OOD2': 'lsun-fix',
                    'OOD3': 'describable-textures'}
    elif args.benchmark == 'imagenet':
        from models.resnet_imagenet import resnet152
        model_fn = resnet152
        num_classes = 200
        last_epoch = 90
        out_data = {'VAL': 'external-imagenet394',
                    'NEAR': 'near-imagenet200',
                    'OOD1': 'food101',
                    'OOD2': 'caltech256',
                    'OOD3': 'places365'}

    if args.method != 'mcdropout' and args.droprate != 0.0:
        raise ValueError('Baseline requires a droprate 0.0.')
    if args.method == 'mcdropout' and args.droprate == 0.0:
        raise ValueError('MCdropout requires a droprate larger than 0.0.')

    # Model initialise
    net = model_fn(num_classes=num_classes, dropRate=args.droprate).to(device)

    if args.method != 'ensemble':
        state_dict = torch.load(f'{args.model_dir}/model_{last_epoch}.pth')
        try:
            net.load_state_dict(state_dict)
        except RuntimeError:
            net.module.load_state_dict(state_dict)

    # Set criterion
    criterion = nn.CrossEntropyLoss().to(device)
    metric_logger = utils.Logger(os.path.join(args.save_dir, 'scores.log'))

    # Search the optimal hyperparameters
    if args.method == 'odin':
        from methods.odin import opt_params
        best_magnitude, best_temperature = opt_params(out_data['VAL'], net, criterion, args)
        params = (best_magnitude, best_temperature)
    elif args.method == 'openmax':
        from methods.openmax.test import opt_params
        mavs, dists, best_alpha, best_eta = opt_params(out_data['VAL'], net, criterion, args)
        params = ((mavs, dists), (best_alpha, best_eta))
    else:
        params = (0, 0)


    ''' Misclassification Detection '''
    md_scores = md_metrics(net, criterion, args, params=params)
    utils.log_record(metric_logger, md_scores, task='miscls')


    ''' Out-of-Distribution Detection '''
    for key in out_data.keys():
        if key == 'VAL':
            mode = 'valid'
        else:
            mode = 'test'
        ood_scores = ood_metrics(out_data[key],
                                 mode,
                                 net,
                                 criterion,
                                 args,
                                 params=params)

        utils.log_record(metric_logger, ood_scores,
                         task=f"{mode}-{out_data[key]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unknown Detection: Test')

    parser.add_argument('--batch-size', default=128, type=int,
                        help='batch size (default: 128)')
    parser.add_argument('--droprate', default=0.0,
                        type=float,
                        help='droprate (default: 0.0)')
    parser.add_argument('--benchmark', default='cifar',
                        type=str,
                        choices=['cifar', 'imagenet'],
                        help='benchmark choice (default: cifar)')
    parser.add_argument('--method', default='baseline',
                        type=str,
                        help='methodology choice (default: baseline)')
    parser.add_argument('--data-dir',
                        default='./data-dir/',
                        type=str, help='data path')
    parser.add_argument('--model-dir', default='./',
                        type=str,
                        help='model path')
    parser.add_argument('--save-dir', default='./save-dir/',
                        type=str,
                        help='save path')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N',
                        help='print frequency (default: 10)')

    args = parser.parse_args()

    main(args)