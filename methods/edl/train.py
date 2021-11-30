import time

import torch

from methods.edl.edl_loss import exp_evidence
from utils import one_hot_embedding, AverageMeter, accuracy


def train(loader, model, criterion, optimizer, epoch, logger, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    if args.benchmark == 'cifar': num_classes = 40
    elif args.benchmark == 'imagenet': num_classes = 200

    model.train()

    end = time.time()
    for i, (inputs, labels) in enumerate(loader):
        data_time.update(time.time() - end)

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        y = one_hot_embedding(labels, num_classes)
        y = y.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, y.float(), epoch, num_classes, args.annealing)

        prec, correct = accuracy(outputs, labels)
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec.item(), inputs.size(0))

        loss.backward()
        optimizer.step()

        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
        acc = torch.mean(match)
        evidence = exp_evidence(outputs)
        alpha = evidence + 1
        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

        total_evidence = torch.sum(evidence, 1, keepdim=True)
        mean_evidence = torch.mean(total_evidence)
        mean_evidence_succ = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
        mean_evidence_fail = torch.sum(
            torch.sum(evidence, 1, keepdim=True) * (1-match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
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

    if args.benchmark == 'cifar': num_classes = 40
    elif args.benchmark == 'imagenet': num_classes = 200

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            data_time.update(time.time() - end)

            inputs, labels = inputs.to(device), labels.to(device)

            y = one_hot_embedding(labels, num_classes)
            y = y.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, y.float(), epoch, num_classes, args.annealing)

            prec, correct = accuracy(outputs, labels)
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec.item(), inputs.size(0))

            match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
            acc = torch.mean(match)
            evidence = exp_evidence(outputs)
            alpha = evidence + 1
            u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

            total_evidence = torch.sum(evidence, 1, keepdim=True)
            mean_evidence = torch.mean(total_evidence)
            mean_evidence_succ = torch.sum(
                torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
            mean_evidence_fail = torch.sum(
                torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                    epoch, i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

    logger.write([epoch, losses.avg, top1.avg])
