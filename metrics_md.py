from __future__ import print_function
import torch
import numpy as np
from sklearn import metrics

from utils import get_values, get_posterior
from dataloader import tst_loader

# get scores on misclassification detection
def md_metrics(net, criterion, args, params=(0,0)):

    print('')
    print('Misclassification Detection')
    print(f'benchmark: {args.benchmark}')
    print('')

    method = args.method

    loader = tst_loader(args.data_dir, args.benchmark, args.batch_size, mode='test')

    if method in ['baseline', 'crl', 'edl', 'oe', 'augmix']:
        _, correct, logit, scores = get_values(loader,
                                               net,
                                               criterion)
    elif method == 'ensemble':
        from methods.ensemble import ensemble
        correct, scores = ensemble(loader,
                                   net,
                                   criterion,
                                   args)
    elif method == 'mcdropout':
        from methods.mcdropout import mcdropout
        correct, scores = mcdropout(loader,
                                    net,
                                    criterion)
    
    elif method == 'odin':
        scores, correct, test_softmax = get_posterior(net,
                                                      loader,
                                                      params[0],
                                                      params[1])
        _, correct, _, _ = get_values(loader,
                                      net,
                                      criterion)
    
    elif method == 'openmax':
        from methods.openmax.openmax import test
        scores1, correct = test(net, loader, params[0], params[1], args.benchmark)
        scores = 1-np.array(scores1)
        correct = correct.astype(int)

    print('-----------------------')
    # acc
    acc = len(np.where(np.array(correct) == 1.0)[0]) / len(correct)
    print(f'* ACC\t\t{round(acc * 100, 2)}')

    # aurc, e-aurc
    conf_corr = sorted(zip(scores, correct), key=lambda x: x[0], reverse=True)
    sorted_conf, sorted_corr = zip(*conf_corr)
    aurc, eaurc = aurc_eaurc(sorted_conf, sorted_corr)

    print('-----------------------')

    md_scores = {'ACC': acc * 100,
                 'AURC': aurc * 1000,
                 'E-AURC': eaurc * 1000}

    return md_scores


# aurc, e-aurc
def aurc_eaurc(rank_conf, rank_corr):
    li_risk = []
    li_coverage = []
    risk = 0
    for i in range(len(rank_conf)):
        coverage = (i + 1) / len(rank_conf)
        li_coverage.append(coverage)

        if rank_corr[i] == 0:
            risk += 1

        li_risk.append(risk / (i + 1))

    r = li_risk[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in li_risk:
        risk_coverage_curve_area += risk_value * (1 / len(li_risk))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print(f'* AURC\t\t{round(aurc*1000, 2)}')
    print(f'* E-AURC\t{round(eaurc*1000, 2)}')

    return aurc, eaurc