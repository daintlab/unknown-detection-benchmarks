# import metrics_md
from utils import accuracy

import numpy as np
import torch
import torch.nn.functional as F

# history update
def history_update(loader, model, history, epoch):
    with torch.no_grad():
        for i, (input, target, idx) in enumerate(loader):
            # set input ,target
            input, target = input.cuda(), target.cuda()
            # compute output
            output = model(input)
            prec, correct = accuracy(output, target)
            #history_update
            history.update(idx, correct, output)

        history.correctness_counting(epoch)

# rank target entropy. -
def rank_target_entropy(data, normalize=False, max_value=None):
    softmax = F.softmax(data, dim=1)
    log_softmax =  F.log_softmax(data, dim=1)
    entropy = softmax * log_softmax
    entropy = -1.0 * entropy.sum(dim=1)
    # normalize [0 ~ 1]
    if normalize:
        normalized_entropy = entropy / max_value
        return -normalized_entropy

    return -entropy

# collect correctness
class History(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.correctness_eaurc = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.correctness_count = 1

    def update(self, data_idx, correctness, output):
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()
        self.correctness_eaurc[data_idx] = correctness.cpu().numpy()

    # counting correctness
    def correctness_counting(self, epoch):
        if epoch > 1:
            self.correctness_count += 1

    # sum correctness
    def get_sum_correctness(self):
        sum_correctness = self.correctness[:]

        return sum_correctness
    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data, total_epochs):
        data_min = self.correctness.min()
        data_max = float(self.correctness_count)

        #return (data - data_min) / (data_max - data_min)
        return data / total_epochs
        # return data

    def rank_target(self, data_idx, data_idx2, kappa_i, kappa_j, total_epochs):
        data_idx = data_idx.cpu().numpy()
        cum_correctness = self.correctness[data_idx]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness = self.correctness_normalize(cum_correctness, total_epochs)
        cum_correctness2 = self.correctness_normalize(cum_correctness2, total_epochs)
        # make target pair
        n_pair = len(data_idx)
        correct_i = cum_correctness[:n_pair]
        correct_j = cum_correctness2[:n_pair]

        # calc target
        target = np.ones(n_pair)

        # detach kappa
        kappa_i = kappa_i.cpu().detach().numpy()
        kappa_j = kappa_j.cpu().detach().numpy()

        # if c_i == c_j and k_i > k_j
        res_bool = np.array((correct_i == correct_j) & (kappa_i > kappa_j))
        target[np.where(res_bool == True)] = -1

        # if c_i < c_j and k_i > k_j
        res_bool = np.array((correct_i < correct_j) & (kappa_i > kappa_j))
        target[np.where(res_bool == True)] = -1

        # if c_i < c_j and k_i < k_j
        res_bool = np.array((correct_i < correct_j) & (kappa_i < kappa_j))
        target[np.where(res_bool == True)] = -1

        target = torch.from_numpy(target).float().cuda()
        # calc margin
        margin = abs(correct_i - correct_j)
        # e^margin
        # margin = np.exp(margin)-1
        margin = torch.from_numpy(margin).float().cuda()

        return target, margin

    # calc eaurc
    # def EAURC(self):
    #     conf_correct = sorted(zip(self.confidence[:], self.correctness_eaurc[:]),
    #                           key=lambda x:x[0], reverse=True)
    #     sorted_conf, sorted_correct = zip(*conf_correct)
    #     aurc, eaurc = metrics_md.aurc_eaurc(sorted_conf, sorted_correct)
    #
    #     return aurc, eaurc
