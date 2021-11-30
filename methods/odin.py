from metrics_ood import ood_metrics

def opt_params(val_data, net ,criterion, args):
    li_temp = [1, 10, 100, 1000]
    li_mag = [0, 0.0005, 0.001, 0.0014, 0.002, 0.0024, 0.005, 0.01, 0.05, 0.1, 0.2]

    odin_best_auroc = 0

    for temperature in li_temp:
        for magnitude in li_mag:
            ood_scores = ood_metrics(val_data,
                                     'valid',
                                     net,
                                     criterion,
                                     args,
                                     params=(magnitude, temperature))

            print(f"temperature: {temperature}  magnitude: {magnitude}  AUROC: {ood_scores['AUROC']}")
            print('')

            if odin_best_auroc < ood_scores['AUROC']:
                odin_best_auroc = ood_scores['AUROC']
                best_temperature = temperature
                best_magnitude = magnitude

    return best_magnitude, best_temperature