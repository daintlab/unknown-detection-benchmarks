import os
import numpy as np

import torch

from utils import get_values

def ensemble(loader, network, criterion, args):
    li_softmax = []
    with torch.no_grad():
        for trial in ['01', '02', '03', '04', '05']:
            if args.benchmark == 'cifar':
                last_epoch = 200
            elif args.benchmark == 'imagenet':
                last_epoch = 90

            state_dict = torch.load(os.path.join(f'{args.model_dir}-{trial}',
                                                 f'model_{last_epoch}.pth'))
            network.load_state_dict(state_dict)
            softmax, correct, logit, _ = get_values(loader,
                                                    network,
                                                    criterion)
            li_softmax.append(softmax)

        li_softmax = np.array(li_softmax)
        softmax_mean = li_softmax.mean(axis=0)

        li_conf = np.max(softmax_mean, 1)
        predicted = np.argmax(softmax_mean, 1)

        acc = 0
        li_correct = []
        for idx, pred_class in enumerate(predicted):
            if pred_class == loader.dataset.targets[idx]:
                acc += 1
                cor = 1
            else:
                cor = 0
            li_correct.append(cor)

    return li_correct, li_conf