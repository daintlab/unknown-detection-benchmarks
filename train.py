import os
import argparse

import dataloader
from utils import Logger
from plot import training_plot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR


def main(args):
    cudnn.benchmark = True

    # Check whether cuda is available
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Create save directory
    save_dir = os.path.join(args.save_dir, f'{args.method}-{args.trial}')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Set train parameters according to the benchmark choice
    if args.benchmark == 'cifar':
        from models.resnet_cifar import resnet20
        num_classes = 40
        model_fn = resnet20
        wd = 0.0005
        nesterov = True
        epochs = 200
        milestones = [120, 160]

    elif args.benchmark == 'imagenet':
        from models.resnet_imagenet import resnet152
        num_classes = 200
        model_fn = resnet152
        wd = 0.0001
        nesterov = False
        epochs = 90
        milestones = [30, 60]

    # Make Data set
    trn_loader = dataloader.trn_loader(args.data_dir,
                                       args.benchmark,
                                       args.batch_size)
    tst_loader = dataloader.tst_loader(args.data_dir,
                                       args.benchmark,
                                       args.batch_size,
                                       mode='test')

    # Set criterion
    criterion = nn.CrossEntropyLoss().to(device)

    # Settings according to each method
    if args.method in ['baseline', 'mcdropout']:
        from methods.baseline.train import train, eval
        if args.method != 'mcdropout' and args.droprate != 0.0:
            raise ValueError('Baseline requires a droprate 0.0.')
        if args.method == 'mcdropout' and args.droprate == 0.0:
            raise ValueError('MCdropout requires a droprate larger than 0.0.')

    elif args.method == 'crl':
        from methods.crl.train import train, eval
        from methods.crl.crl_loader import loader
        from methods.crl.crl_utils import History
        trn_loader, tst_loader = loader(args.data_dir,
                                        args.benchmark,
                                        args.batch_size)
        cls_criterion = nn.CrossEntropyLoss().to(device)
        rank_criterion = nn.MarginRankingLoss(margin=0.0,
                                              reduction='none').to(device)

        trn_history = History(len(trn_loader.dataset))
        tst_history = History(len(tst_loader.dataset))

        criterion = [cls_criterion, rank_criterion, trn_history, tst_history]

    elif args.method == 'augmix':
        from methods.augmix.train import train, eval
        from methods.augmix.augmix_loader import loader
        trn_loader, tst_loader = loader(args.data_dir,
                                        args.benchmark,
                                        args.batch_size)

    elif args.method == 'edl':
        from methods.edl.train import train, eval
        from methods.edl.edl_loss import edl_log_loss
        criterion = edl_log_loss

    elif args.method == 'oe':
        from methods.oe.train import train, eval
        from methods.oe.oe_loader import trn_out_loader
        trn_out_loader = trn_out_loader(args.data_dir,
                                        args.benchmark,
                                        2*args.batch_size)
        trn_loader = (trn_loader, trn_out_loader)


    # Model initialise
    net = nn.DataParallel(model_fn(num_classes=num_classes, dropRate=args.droprate)).to(device)

    # Set optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=wd,
                          nesterov=nesterov)

    # Set scheduler
    scheduler = MultiStepLR(optimizer,
                            milestones=milestones,
                            gamma=0.1)

    # Make logger
    trn_logger = Logger(os.path.join(save_dir, 'trn.log'))
    tst_logger = Logger(os.path.join(save_dir, 'tst.log'))

    # Start training the model
    for epoch in range(1, epochs+1):
        # Train
        print('\n*--- Train')
        train(trn_loader,
              net,
              criterion,
              optimizer,
              epoch,
              trn_logger,
              args,
              device)
        # Evaluation
        print('\n*--- Test')
        eval(tst_loader,
             net,
             criterion,
             epoch,
             tst_logger,
             args,
             device)
        # scheduler
        scheduler.step()
    # Finish train

    # Save the last model
    torch.save(net.state_dict(),
               os.path.join(save_dir, f'model_{int(epochs)}.pth'))

    # Draw plot
    training_plot(trn_logger, tst_logger, save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Unknown Detection: Train a Classifier')

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
                        choices=['baseline', 'crl', 'augmix', 'edl', 'oe'],
                        help='methodology choice (default: baseline)')
    parser.add_argument('--data-dir',
                        default='./data-dir/',
                        type=str, help='data path')
    parser.add_argument('--save-dir', default='./save-dir/', type=str,
                        help='save path')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--trial', default='01', type=str)

    # edl
    parser.add_argument("--annealing",
                        default=None,
                        type=float,
                        help='an annealing coefficient for edl')

    args = parser.parse_args()

    main(args)