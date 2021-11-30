from dataloader import trn_loader
from metrics_ood import ood_metrics
from methods.openmax.openmax import compute_train_score_and_mavs_and_dists


def opt_params(val_data, net ,criterion, args):

    if args.benchmark == 'cifar': num_classes = 40
    elif args.benchmark == 'imagenet': num_classes = 200

    ''' validation '''
    print('')
    print('Validation')
    print(f'known data: {args.benchmark}')
    print(f"unknown data: {val_data}")
    print('')

    train_loader = trn_loader(args.data_dir,
                              args.benchmark,
                              args.batch_size)

    # Fit the weibull distribution from training data.
    _, mavs, dists = compute_train_score_and_mavs_and_dists(
        num_classes, train_loader, net, eu_weight=5e-3)

    li_alpha = [1, 5, 10, 20, 30, 40]
    li_eta = [2, 5, 10, 20, 30, 40]

    best_auroc = 0
    for alpha in li_alpha:
        for eta in li_eta:
            params = ((mavs, dists), (alpha, eta))

            ood_scores = ood_metrics(val_data,
                                 'valid',
                                 net,
                                 criterion,
                                 args,
                                 params=params)

            print(f"alpha: {alpha}  eta: {eta}  AUROC: {ood_scores['AUROC']}")
            print('')

            if best_auroc < ood_scores['AUROC']:
                best_auroc = ood_scores['AUROC']
                best_alpha = alpha
                best_eta = eta
    print(f'the opt. alpha is {best_alpha}, the opt. eta is {best_eta}')

    return mavs, dists, best_alpha, best_eta