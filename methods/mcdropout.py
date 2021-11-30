import torch
import numpy as np
from utils import get_values


def mcdropout(loader, network, criterion):
    li_softmax = []
    li_logit = []

    mc_times = 50
    network.eval()
    network.training = True
    with torch.no_grad():
        for time in range(1, mc_times+1):
            print(f'* mc dropout {time} / {mc_times}')
            softmax, correct, logit, confidence = get_values(loader,
                                                             network,
                                                             criterion)
            li_softmax.append(softmax)
            li_logit.append(logit)

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