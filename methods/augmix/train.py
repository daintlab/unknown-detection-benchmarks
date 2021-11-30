import time

from utils import AverageMeter, accuracy

import torch
import torch.nn.functional as F


def train(loader, model, criterion, optimizer, epoch, logger, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.train()
    for i, (images, targets) in enumerate(loader):
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        images_all = torch.cat(images, 0).to(device)
        targets = targets.to(device)
        logits_all = model(images_all)
        logits_clean, logits_aug1, logits_aug2 = torch.split(
            logits_all, images[0].size(0))

        # Cross-entropy is only computed on clean images
        loss = criterion(logits_clean, targets)

        p_clean, p_aug1, p_aug2 = F.softmax(logits_clean, dim=1), \
                                  F.softmax(logits_aug1, dim=1), \
                                  F.softmax(logits_aug2, dim=1)

        # Clamp mixture distribution to avoid exploding KL divergence
        p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
        loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                      F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                      F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.

        prec, correct = accuracy(logits_clean, targets)
        losses.update(loss.item(), len(images[0]))
        top1.update(prec.item(), len(images[0]))

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

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(loader):
            data_time.update(time.time() - end)

            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            loss = criterion(logits, targets)

            prec, correct = accuracy(logits, targets)
            losses.update(loss.item(), images.size(0))
            top1.update(prec.item(), images.size(0))

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

