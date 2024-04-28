from __future__ import division, print_function
import numpy as np
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
from scipy.optimize import linear_sum_assignment as linear_assignment
import random
import os
import argparse
from utils.faster_mix_k_means_pytorch import K_Means as SemiSupKMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
from sklearn.cluster import KMeans
import logging
from faiss import Kmeans as faiss_Kmeans

import time
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from typing import List



# -------------------------------
# Evaluation Criteria
# -------------------------------
def evaluate_clustering(y_true, y_pred):

    start = time.time()
    # print('Computing metrics...')
    if len(set(y_pred)) < 1000:
        acc = cluster_acc(y_true.astype(int), y_pred.astype(int))
    else:
        acc = None

    nmi = nmi_score(y_true, y_pred)
    ari = ari_score(y_true, y_pred)
    pur = purity_score(y_true, y_pred)
    # print(f'Finished computing metrics {time.time() - start}...')

    return acc, nmi, ari, pur


def cluster_acc(y_pred, y_true, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T
    #numpy.vstack() function is used to stack the sequence of input arrays vertically to make a single array.

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

# -------------------------------
# Mixed Eval Function
# -------------------------------
def mixed_eval(targets, preds, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)

    # Labelled examples
    if mask.sum() == 0:  # All examples come from unlabelled classes

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int), preds.astype(int)), \
                                                         nmi_score(targets, preds), \
                                                         ari_score(targets, preds)

        print('Unlabelled Classes Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'
              .format(unlabelled_acc, unlabelled_nmi, unlabelled_ari))

        # Also return ratio between labelled and unlabelled examples
        return (unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()

    else:

        labelled_acc, labelled_nmi, labelled_ari = cluster_acc(targets.astype(int)[mask],
                                                               preds.astype(int)[mask]), \
                                                   nmi_score(targets[mask], preds[mask]), \
                                                   ari_score(targets[mask], preds[mask])

        unlabelled_acc, unlabelled_nmi, unlabelled_ari = cluster_acc(targets.astype(int)[~mask],
                                                                     preds.astype(int)[~mask]), \
                                                         nmi_score(targets[~mask], preds[~mask]), \
                                                         ari_score(targets[~mask], preds[~mask])

        # Also return ratio between labelled and unlabelled examples
        return (labelled_acc, labelled_nmi, labelled_ari), (
            unlabelled_acc, unlabelled_nmi, unlabelled_ari), mask.mean()


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x


class BCE(nn.Module):
    eps = 1e-7 # Avoid calculating log(0). Use the small value of float16.
    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1)==len(prob2)==len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),str(len(prob2)),str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


def PairEnum(x,mask=None):

    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))

    if mask is not None:

        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        #dim 0: #sample, dim 1:#feature 
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))

    return x1, x2


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc


def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc


EVAL_FUNCS = {
    'v1': split_cluster_acc_v1,
    'v2': split_cluster_acc_v2,
}

def log_accs_from_preds(y_true, y_pred, mask, eval_funcs: List[str], save_name: str, T: int=None, writer: SummaryWriter=None,
                        print_output=False):

    """
    Given a list of evaluation functions to use (e.g ['v1', 'v2']) evaluate and log ACC results

    :param y_true: GT labels
    :param y_pred: Predicted indices
    :param mask: Which instances belong to Old and New classes
    :param T: Epoch
    :param eval_funcs: Which evaluation functions to use
    :param save_name: What are we evaluating ACC on
    :param writer: Tensorboard logger
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    for i, f_name in enumerate(eval_funcs):

        acc_f = EVAL_FUNCS[f_name]
        all_acc, old_acc, new_acc = acc_f(y_true, y_pred, mask)
        log_name = f'{save_name}_{f_name}'

        if writer is not None:
            writer.add_scalars(log_name,
                               {'Old': old_acc, 'New': new_acc,
                                'All': all_acc}, T)

        if i == 0:
            to_return = (all_acc, old_acc, new_acc)

        if print_output:
            print_str = f'Epoch {T}, {log_name}: All {all_acc:.4f} | Old {old_acc:.4f} | New {new_acc:.4f}'
            print(print_str)

    return to_return

@torch.no_grad()
def test_kmeans(model, test_loader,
                epoch, save_name,
                args):

    model.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label) in enumerate(tqdm(test_loader)):

        images = images.cuda()

        # Pass features through base model and then additional learnable transform (linear layer)
        feats = model(images)

        feats = torch.nn.functional.normalize(feats, dim=-1)

        all_feats.append(feats.cpu().numpy())
        targets = np.append(targets, label.cpu().numpy())
        mask = np.append(mask, np.array([True if x.item() in range(len(args.train_classes))
                                         else False for x in label]))

    # -----------------------
    # K-MEANS
    # -----------------------
    print('Fitting K-Means...')
    all_feats = np.concatenate(all_feats)
    kmeans = KMeans(n_clusters=args.num_labeled_classes + args.num_unlabeled_classes, random_state=0).fit(all_feats)
    preds = kmeans.labels_
    print('Done!')

    # -----------------------
    # EVALUATE
    # -----------------------
    all_acc, old_acc, new_acc = log_accs_from_preds(y_true=targets, y_pred=preds, mask=mask,
                                                    T=epoch, eval_funcs=args.eval_funcs, save_name=save_name,
                                                    writer=args.writer)

    return all_acc, old_acc, new_acc

@torch.no_grad()
def test_kmeans_mix(G, mlp, s_feats, s_labels, test_loader, k, args, logger):

    G.eval()
    mlp.eval()

    s_feats = F.normalize(s_feats, dim=-1).cpu()
    s_labels = s_labels.cpu()
    t_feats = []
    t_labels = np.array([])

    for i, (img, label, idx) in enumerate(test_loader):

        img = img.cuda()
        feats = mlp(G(img))
        feats = F.normalize(feats, dim=-1)
        t_feats.append(feats.cpu())
        t_labels = np.append(t_labels, label.cpu())
    t_feats = torch.cat(t_feats)

    print('Fitting Semi-Supervised K-Means...')

    kmeans = SemiSupKMeans(k=k, tolerance=1e-4, max_iterations=args.max_kmeans_iter, init='k-means++',
                        n_init=args.k_means_init, random_state=None, n_jobs=None, pairwise_batch_size=1024, mode=None)

    kmeans.fit_mix(t_feats, s_feats, s_labels)
    all_preds = kmeans.labels_.cpu().numpy()
    s_num = len(s_feats)
    t_preds = all_preds[s_num:]
    # np.set_printoptions(threshold=np.inf)
    # print(all_preds)

        # the number of all items which are predicted correctly
    correct_all, size_all = 0, 0
        # the number of known items which are predicted correctly
    correct_kno, size_kno = 0, 0
        # the number of unknown items which are predicted correctly 
        # when all unknown classes are regarded as a whole unknown class
    correct_unk_whole, size_unk = 0, 0
    correct_c_in_common = 0
    clus_pred, clus_GT = [],  []# for clustering acc

    n_share = args.conf.data.dataset.n_share
    n_source_private = args.conf.data.dataset.n_source_private
    n_total = args.conf.data.dataset.n_total
    common_classes = range(n_share)
    t_private_classes = range(n_share+n_source_private, n_total)

    for pred_lbl, GT in zip(t_preds, t_labels):
        if GT in common_classes:
            correct_kno += (int)(pred_lbl == GT)
            if pred_lbl in common_classes:
                correct_c_in_common += 1
            size_kno += 1
        elif GT in t_private_classes:
            correct_unk_whole += (int)(pred_lbl in t_private_classes)
            size_unk += 1
            clus_pred.append(pred_lbl)
            clus_GT.append(GT)
        size_all += 1

    assert size_all == size_kno + size_unk

    clus_pred = np.array(clus_pred)
    clus_GT = np.array(clus_GT)

    correct_all += correct_kno + correct_unk_whole
    acc_c_in_c = correct_c_in_common / size_kno
    acc_c = correct_kno / size_kno
    acc_t = correct_unk_whole / size_unk
    acc_all = correct_all / size_all
    if (acc_c + acc_t) == 0:
        h_score = 0
    else:
        h_score = 2 * acc_c * acc_t / (acc_c + acc_t)


    
    t_unk_clus_acc, t_unk_clus_nmi, t_unk_clus_ari, t_unk_clus_pur = evaluate_clustering(clus_GT, clus_pred)
    logger.setLevel(logging.INFO)
    logger.info(
            f"acc_all = {correct_all}/{size_all}({100. * acc_all:.3f}%) h_score = {100. * h_score:.3f}%\n"
            f"acc_c_in_c = {correct_c_in_common}/{size_kno}({100. * acc_c_in_c:.3f}%) acc_kno = {correct_kno}/{size_kno}({100. * acc_c:.3f}%) acc_unk = {correct_unk_whole}/{size_unk}({100. * acc_t:.3f}%)" 
            f"[t_unk_clus] acc = {t_unk_clus_acc:.3f} nmi = {t_unk_clus_nmi:.3f} ari = {t_unk_clus_ari:.3f} pur = {t_unk_clus_pur:.3f}")

    return h_score, acc_c, acc_t, acc_all, acc_c_in_c, t_unk_clus_acc, t_unk_clus_nmi, t_unk_clus_ari, t_unk_clus_pur

    
@torch.no_grad()
def test_kmeans_mix_search_k(G, mlp, s_feats, s_labels, t_feats, k_begin, k_end, args, logger):
    print(f"searching from {k_begin} to {k_end}")
    logger.setLevel(logging.INFO)
    G.eval()
    mlp.eval()

    s_feats = F.normalize(s_feats, dim=-1).cpu()
    t_feats = F.normalize(t_feats, dim=-1).cpu()
    s_labels = s_labels.cpu()
    all_feat = torch.cat((s_feats, t_feats)).numpy()

    best_k = k_begin
    best_acc = 0
    print('Searching best k...')
    d =  s_feats.shape[-1]
    for k in tqdm(range(k_begin, k_end+1)):
        kmeans = faiss_Kmeans(
                d,
                k, # number of clusters
                niter=40,
                verbose=False,
                spherical=True,
                min_points_per_centroid=1,
                max_points_per_centroid=10000,
                gpu=False,
                seed=k+1337,
                frozen_centroids=False,
            )
        kmeans.train(all_feat, init_centroids=None)
        _, I = kmeans.index.search(all_feat, 1)
        s_num = len(s_feats)
        s_preds = I.squeeze(1)[:s_num]
        # np.set_printoptions(threshold=np.inf)
        # print(all_preds)


        s_preds = np.array(s_preds)
        s_GT = np.array(s_labels)
        acc, nmi, ari, pur = evaluate_clustering(s_GT, s_preds)
        # logger.info(f"k = {k}, acc = {acc:.3f}")
        if acc > best_acc:
            best_k = k
            best_acc = acc

    logger.info(f"best_k = {best_k}, best_acc = {best_acc:.3f}\n")
           

    return best_k, best_acc

    
