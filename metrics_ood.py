from __future__ import print_function

import numpy as np
from sklearn import metrics

from dataloader import in_dist_loader, out_dist_loader
from utils import get_values, get_posterior


def get_curve(in_scores, out_scores, stypes=['Baseline']):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()
    for stype in stypes:
        known = in_scores
        novel = out_scores

        known.sort()
        novel.sort()
        end = np.max([np.max(known), np.max(novel)])
        start = np.min([np.min(known), np.min(novel)])
        num_k = known.shape[0]
        num_n = novel.shape[0]
        tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
        tp[stype][0], fp[stype][0] = num_k, num_n
        k, n = 0, 0
        for l in range(num_k + num_n):
            if k == num_k:
                tp[stype][l + 1:] = tp[stype][l]
                fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
                break
            elif n == num_n:
                tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
                fp[stype][l + 1:] = fp[stype][l]
                break
            else:
                if novel[n] < known[k]:
                    n += 1
                    tp[stype][l + 1] = tp[stype][l]
                    fp[stype][l + 1] = fp[stype][l] - 1
                else:
                    k += 1
                    tp[stype][l + 1] = tp[stype][l] - 1
                    fp[stype][l + 1] = fp[stype][l]
        tpr95_pos = np.abs(tp[stype] / num_k - .95).argmin()
        fpr_at_tpr95[stype] = fp[stype][tpr95_pos] / num_n

    return tp, fp, fpr_at_tpr95



def ood_metrics(out_data, mode, net, criterion, args, stypes=['result'], params=(0,0)):

    print('')
    print('Near/Far-OoD Detection')
    print(f'known data: {args.benchmark}')
    print(f"unknown data: {out_data}")
    print('')

    method = args.method

    in_loader = in_dist_loader(args.data_dir,
                               args.benchmark,
                               args.batch_size,
                               mode)
    out_loader = out_dist_loader(args.data_dir,
                                 args.benchmark,
                                 out_data,
                                 args.batch_size,
                                 mode)

    if method in ['baseline', 'crl', 'edl', 'augmix', 'oe']:
        _, _, _, in_scores = get_values(in_loader,
                                        net,
                                        criterion)
        _, _, _, out_scores = get_values(out_loader,
                                         net,
                                         criterion)
    elif method == 'ensemble':
        from methods.ensemble import ensemble
        _, in_scores = ensemble(in_loader,
                                net,
                                criterion,
                                args)
        _, out_scores = ensemble(out_loader,
                                 net,
                                 criterion,
                                 args)
    elif method == 'mcdropout':
        from methods.mcdropout import mcdropout
        _, in_scores = mcdropout(in_loader,
                                 net,
                                 criterion)
        _, out_scores = mcdropout(out_loader,
                                  net,
                                  criterion)
                                  
    elif method == 'odin':
        print(f'temperature: {params[1]}  magnitude: {params[0]}')
        in_scores, _, _ = get_posterior(net,
                                        in_loader,
                                        params[0],
                                        params[1])

        out_scores, _, _ = get_posterior(net,
                                         out_loader,
                                         params[0],
                                         params[1])
        
    elif method == 'openmax':
        from methods.openmax.openmax import test
        in_openmax, _ = test(net,
                             in_loader,
                             params[0],
                             params[1],
                             args.benchmark)
        out_openmax, _ = test(net,
                              out_loader,
                              params[0],
                              params[1],
                              args.benchmark)

        in_scores = 1-np.array(in_openmax)
        out_scores = 1-np.array(out_openmax)

    tp, fp, fpr_at_tpr95 = get_curve(in_scores, out_scores, stypes)
    results = dict()

    for stype in stypes:
        results = dict()

        # FPR@95%TPR
        mtype = 'FPR@95%TPR'
        results[mtype] = fpr_at_tpr95[stype] * 100

        # AUROC
        mtype = 'AUROC'
        tpr = np.concatenate([[1.], tp[stype] / tp[stype][0], [0.]])
        fpr = np.concatenate([[1.], fp[stype] / fp[stype][0], [0.]])
        results[mtype] = -np.trapz(1. - fpr, tpr)  * 100

    print('-----------------------')
    print(f"* FPR@95%TPR\t{round(results['FPR@95%TPR'], 2)}")
    print(f"* AUROC\t\t{round(results['AUROC'], 2)}")
    print('-----------------------')

    return results