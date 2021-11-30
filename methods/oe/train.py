import time

import torch
from utils import AverageMeter, accuracy

def train(loaders, model, criterion, optimizer, epoch, logger, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.train()  # enter train mode
    for i, (in_set, out_set) in enumerate(zip(loaders[0], loaders[1])):
        data_time.update(time.time() - end)
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]
        data, target = data.to(device), target.to(device)
        x = model(data)

        # backward
        optimizer.zero_grad()
        loss = criterion(x[:len(in_set[0])], target)
        loss += 0.5 * - (x[len(in_set[0]):].mean(1) - torch.logsumexp(
            x[len(in_set[0]):], dim=1)).mean()

        loss.backward()
        optimizer.step()

        prec, correct = accuracy(x[:target.shape[0]], target)
        losses.update(loss.item(), in_set[0].size(0))
        top1.update(prec.item(), in_set[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                epoch, i, len(loaders[0]), batch_time=batch_time,
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
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            prec, correct = accuracy(output, target)
            losses.update(loss.item(), data.size(0))
            top1.update(prec.item(), data.size(0))

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