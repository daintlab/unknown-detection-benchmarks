import numpy as np
import scipy.spatial.distance as spd
import torch

import libmr


def calc_distance(query_score, mcv, eu_weight, distance_type='eucos'):
    if distance_type == 'eucos':
        query_distance = spd.euclidean(mcv, query_score) * eu_weight + spd.cosine(mcv, query_score)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mcv, query_score)
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mcv, query_score)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance


def fit_weibull(means, dists, categories, tailsize=20, distance_type='eucos'):
    """
    Input:
        means (C, channel, C)
        dists (N_c, channel, C) * C
    Output:
        weibull_model : Perform EVT based analysis using tails of distances and save
                        weibull model parameters for re-adjusting softmax scores
    """
    weibull_model = {}
    for mean, dist, category_name in zip(means, dists, categories):
        weibull_model[category_name] = {}
        weibull_model[category_name]['distances_{}'.format(distance_type)] = dist[distance_type]
        weibull_model[category_name]['mean_vec'] = mean
        weibull_model[category_name]['weibull_model'] = []
        for channel in range(mean.shape[0]):
            mr = libmr.MR()
            tailtofit = np.sort(dist[distance_type][channel, :])[-tailsize:]
            mr.fit_high(tailtofit, len(tailtofit))
            weibull_model[category_name]['weibull_model'].append(mr)

    return weibull_model


def query_weibull(category_name, weibull_model, distance_type='eucos'):
    return [weibull_model[category_name]['mean_vec'],
            weibull_model[category_name]['distances_{}'.format(distance_type)],
            weibull_model[category_name]['weibull_model']]


def compute_openmax_prob(scores, scores_u):
    prob_scores, prob_unknowns = [], []
    logit_k, logit_u = [], []
    for s, su in zip(scores, scores_u):
        channel_scores = np.exp(s)
        channel_unknown = np.exp(np.sum(su))
        total_denom = np.sum(channel_scores) + channel_unknown
        prob_scores.append(channel_scores / total_denom)
        prob_unknowns.append(channel_unknown / total_denom)
        logit_k.append(s)
        logit_u.append(su)
    # Take channel mean
    scores = np.mean(prob_scores, axis=0)
    unknowns = np.mean(prob_unknowns, axis=0)
    logit_k_mean = np.mean(logit_k, axis=0)
    logit_u_mean = np.sum(logit_u)
    modified_scores = scores.tolist() + [unknowns]
    logit_scores = logit_k_mean.tolist() + [logit_u_mean]
    return modified_scores, logit_scores

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def openmax(weibull_model, categories, input_score, eu_weight, benchmark, alpha=10, distance_type='eucos'):
    """Re-calibrate scores via OpenMax layer
    Output:
        openmax probability and softmax probability
    """
    nb_classes = len(categories)
    ranked_list = input_score.argsort().ravel()[::-1][:alpha]
    alpha_weights = [0.5]
    if benchmark == 'imagenet':
        alpha_weights = [((alpha + 1) - i) / float(alpha) for i in range(1, alpha + 1)]
    omega = np.zeros(nb_classes)
    omega[ranked_list] = alpha_weights
    scores, scores_u = [], []
    for channel, input_score_channel in enumerate(input_score):
        score_channel, score_channel_u = [], []
        for c, category_name in enumerate(categories):
            mav, dist, model = query_weibull(category_name, weibull_model, distance_type)
            channel_dist = calc_distance(input_score_channel, mav[channel], eu_weight, distance_type)
            wscore = model[channel].w_score(channel_dist)
            modified_score = input_score_channel[c] * (1 - wscore * omega[c])
            score_channel.append(modified_score)
            score_channel_u.append(input_score_channel[c] - modified_score)
        scores.append(score_channel)
        scores_u.append(score_channel_u)
    scores = np.asarray(scores)
    scores_u = np.asarray(scores_u)
    openmax_prob, openmax_logit = np.array(compute_openmax_prob(scores, scores_u))
    softmax_prob = softmax(np.array(input_score.ravel()))
    return openmax_prob, softmax_prob, openmax_logit


def compute_channel_distances(mavs, features, eu_weight=0.5):
    """
    Input:
        mavs (channel, C)
        features: (N, channel, C)
    Output:
        channel_distances: dict of distance distribution from MAV for each channel.
    """
    eucos_dists, eu_dists, cos_dists = [], [], []
    for channel, mcv in enumerate(mavs):  # Compute channel specific distances
        eu_dists.append([spd.euclidean(mcv, feat[channel]) for feat in features])
        cos_dists.append([spd.cosine(mcv, feat[channel]) for feat in features])
        eucos_dists.append([spd.euclidean(mcv, feat[channel]) * eu_weight +
                            spd.cosine(mcv, feat[channel]) for feat in features])

    return {'eucos': np.array(eucos_dists), 'cosine': np.array(cos_dists), 'euclidean': np.array(eu_dists)}


def compute_train_score_and_mavs_and_dists(train_class_num, loader, net, eu_weight):
    scores = [[] for _ in range(train_class_num)]
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            for score, t in zip(outputs, targets):
                if torch.argmax(score) == t:
                    scores[t].append(score.unsqueeze(dim=0).unsqueeze(dim=0))

    scores = [torch.cat(x).cpu().numpy() for x in scores]  # (N_c, 1, C) * C
    mavs = np.array([np.mean(x, axis=0) for x in scores])  # (C, 1, C)
    dists = [compute_channel_distances(mcv, score, eu_weight) for mcv, score in zip(mavs, scores)]

    return scores, mavs, dists

def test(net, loader, param1, param2, benchmark):
    if benchmark == 'cifar': num_classes = 40
    elif benchmark == 'imagenet': num_classes = 200

    mavs, dists = param1
    alpha, eta = param2

    net.eval()

    scores, labels = [], []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            scores.append(outputs)
            labels.append(targets)

    # Get the prdict results.
    scores = torch.cat(scores,dim=0).cpu().numpy()
    labels = torch.cat(labels,dim=0).cpu().numpy()
    scores = np.array(scores)[:, np.newaxis, :]
    labels = np.array(labels)

    categories = list(range(0, num_classes))
    weibull_model = fit_weibull(mavs, dists, categories, eta)

    pred_openmax = []
    li_openmax = []
    
    for score in scores:
        so, ss, ol = openmax(weibull_model, categories, score, 5e-3,
                             benchmark, alpha)  # openmax_prob, softmax_prob

        li_openmax.append(so[num_classes])
        pred_openmax.append(np.argmax(so) if np.max(so) >= 0.0 else num_classes)

    correct = (labels == pred_openmax)
    return li_openmax, correct
