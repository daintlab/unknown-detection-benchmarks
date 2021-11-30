import time

import torch
import torch.nn.functional as F

from utils import AverageMeter, accuracy

def train(loader, model, criterion, optimizer, epoch, logger, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.train()
    for i, (input, target) in enumerate(loader):
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = criterion(output, target)

        # record accuracy and loss
        prec, correct = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

    logger.write([epoch, losses.avg, top1.avg])


def eval(loader, model, criterion, epoch, logger, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            data_time.update(time.time() - end)

            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion(output, target)

            # record accuracy and loss
            prec, correct = accuracy(output, target)
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

        logger.write([epoch, losses.avg, top1.avg])
