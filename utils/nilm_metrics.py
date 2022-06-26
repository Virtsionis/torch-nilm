import torch
import numpy as np
from constants.constants import*


def NILMmetrics(pred: np.array, ground: np.array, threshold: int = 40, round_digit: int = 3):

    def convert_precision_16_to_32(arr):
        return arr.astype(float)

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

    def relative_error_total_energy(predictions, groundtruth):
        e_pred = sum(predictions)
        e_ground = sum(groundtruth)
        return abs(e_pred - e_ground) / float(max(e_pred, e_ground))

    def mean_absolute_error(predictions, groundtruth):
        sum_samples = len(predictions)
        total_sum = sum(abs(predictions - groundtruth))
        return total_sum / sum_samples

    if torch.is_tensor(pred):
        pr = pred.numpy()
    else:
        pr = pred

    if torch.is_tensor(ground):
        gr = ground.numpy()
    else:
        gr = ground

    pr, gr = convert_precision_16_to_32(pred), convert_precision_16_to_32(ground)
    pr[np.isnan(pr)] = 0
    gr[np.isnan(gr)] = 0

    rete = round(relative_error_total_energy(pr, gr), round_digit)
    mae = mean_absolute_error(pr, gr)
    mae = round(mae, round_digit)

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
                       COLUMN_MAE: mae, COLUMN_RETE: rete}
    return metrics_results
