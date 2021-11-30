import time

from utils import AverageMeter, accuracy

import torch
import torch.nn.functional as F

def train(loader, model, criterion, optimizer, epoch, logger, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    cls_losses = AverageMeter()
    ranking_losses = AverageMeter()
    end = time.time()

    if args.benchmark == 'cifar': epochs = 200
    elif args.benchmark == 'imagenet': epochs = 90

    model.train()
    for i, (input, target, idx) in enumerate(loader):
        data_time.update(time.time() - end)

        input, target = input.to(device), target.to(device)
        output = model(input)
        cls_loss = criterion[0](output, target)

        # Compute ranking target value normalize (0 ~ 1) range
        probs = F.softmax(output, dim=1)
        conf, _ = probs.max(dim=1)
        rank_data = conf

        # Make pair
        rank_input1 = rank_data
        rank_input2 = torch.roll(rank_data, -1)

        # Calculate ranking target
        idx2 = torch.roll(idx, -1)
        rank_target, rank_margin = criterion[2].rank_target(idx, idx2,
                                                            rank_input1,
                                                            rank_input2,
                                                            epochs)

        # rank_target \in {-1, 1}
        rank_input2 = rank_input2 + rank_margin / rank_target

        ranking_loss = criterion[1](rank_input1,
                                    rank_input2,
                                    rank_target)

        ranking_loss = ranking_loss.mean()
        loss = cls_loss + ranking_loss

        # measure accuracy and record loss
        prec, correct = accuracy(output, target)
        losses.update(loss.item(), input.size(0))
        cls_losses.update(cls_loss.item(), input.size(0))
        ranking_losses.update(ranking_loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Rank Loss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'
                  'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
                   epoch, i, len(loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, cls_loss=cls_losses,
                   rank_loss=ranking_losses,top1=top1))

        # correctness counting
        criterion[2].update(idx, correct, output)
        
    criterion[2].correctness_counting(epoch)
    
    logger.write([epoch, losses.avg, top1.avg])


def eval(loader, model, criterion, epoch, logger, args, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    model.eval()
    with torch.no_grad():
        for i, (input, target, idx) in enumerate(loader):
            data_time.update(time.time() - end)

            input, target = input.to(device), target.to(device)
            output = model(input)
            loss = criterion[0](output, target)

            # measure accuracy and record loss
            prec, correct = accuracy(output, target)
            criterion[3].update(idx, correct, output)
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
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
