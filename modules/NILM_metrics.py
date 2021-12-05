import torch
import numpy as np
from constants.constants import*


def NILM_metrics(pred: np.array, ground: np.array, threshold: int = 40, mmax: float = None, means: float = None,
                 stds: float = None, round_digit: int = 3):

    def tp_tn_fp_fn(states_pred, states_ground):
        tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
        fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
        fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
        tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
        return tp, tn, fp, fn

    def recall(tp, fn):
        return tp/float(tp+fn)

    def precision(tp, fp):
        return tp/float(tp+fp)

    def f1(prec, rec):
        return 2 * (prec*rec) / float(prec+rec)

    def accuracy(tp, tn, p, n):
        return (tp + tn) / float(p + n)

    def relative_error_total_energy(pred, ground):

        E_pred = sum(pred)
        E_ground = sum(ground)
        return abs(E_pred - E_ground) / float(max(E_pred,E_ground))

    def mean_absolute_error(pred, ground):
        sum_samples = len(pred)
        total_sum = sum(abs((pred) - ground))
        return total_sum / sum_samples

#====================================================================#
#                             "main"                                 #
#====================================================================#

    '''normalize the threshold if must'''
    if mmax:
        threshold = threshold/mmax

    if means and stds:
        threshold = (threshold - means)/stds

    print(f"Threshold {threshold}")

    if torch.is_tensor(pred):
        pr = pred.numpy()
    else:
        pr = pred

    if torch.is_tensor(ground):
        gr = ground.numpy()
    else:
        gr = ground

    pr[np.isnan(pr)] = 0
    gr[np.isnan(gr)] = 0

    RETE = round(relative_error_total_energy(pr, gr),round_digit)
    MAE = mean_absolute_error(pr, gr)

    if mmax:
        MAE *= mmax

    if means and stds:
        MAE *= stds
        # MAE += means
    MAE = round(MAE, round_digit)

    pr = np.array([0 if p < threshold else 1 for p in pr])
    gr = np.array([0 if p < threshold else 1 for p in gr])

    tp, tn, fp, fn = tp_tn_fp_fn(pr, gr)
    positives = sum(pr)
    negatives = len(pr) - positives

    recall = round(recall(tp, fn), round_digit)
    precision = round(precision(tp, fp), round_digit)
    f1 = round(f1(precision,recall), round_digit)
    accuracy = round(accuracy(tp, tn, positives, negatives), round_digit)

    metrics_results = {COLUMN_RECALL: recall, COLUMN_PRECISION: precision,
                       COLUMN_F1: f1, COLUMN_ACCURACY: accuracy,
                       COLUMN_MAE: MAE, COLUMN_RETE: RETE}
    return metrics_results
