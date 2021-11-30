import torch
import torch.nn.functional as F


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))

def kl_divergence(alpha, num_classes):
    beta = torch.ones([1, num_classes], dtype=torch.float32).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step):
    y = y.cuda()
    alpha = alpha.cuda()
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    # annealing_coef = torch.min(torch.tensor(
    #     1.0, dtype=torch.float32), torch.tensor(epoch_num / annealing_step, dtype=torch.float32))

    kl_alpha = (alpha - 1) * (1 - y) + 1
    annealing_coef = annealing_step
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes)
    return A + kl_div

def edl_log_loss(output, target, epoch_num, num_classes, annealing_step):
    evidence = exp_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(edl_loss(torch.log, target, alpha,
                               epoch_num, num_classes, annealing_step))
    return loss