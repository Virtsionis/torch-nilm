import torch
import numpy as np
from numba import njit
from constants.constants import*


def NILMmetrics(pred: np.array, ground: np.array, threshold: int = 40, rounding_digit: int = 3):

    def convert_precision_16_to_32(arr):
        return arr.astype(float)

    @njit
    def get_eac(prediction, target):
        num = np.sum(np.abs(prediction - target))
        den = (np.sum(target))
        eac_ = 1 - (num / den) / 2
        eac_ = np.where(eac_ < 0, 0, eac_)
        return eac_

    @njit
    def get_relative_error(target, prediction):
        return np.mean(np.nan_to_num(np.abs(target - prediction) / np.maximum(target, prediction)))

    @njit
    def get_nde(prediction, target, round_digit=3):
        return round(np.sum((target - prediction) ** 2) / np.sum((target ** 2)), round_digit)

    @njit
    def tp_tn_fp_fn(states_pred, states_ground):
        tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
        fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
        fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
        tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
        return tp, tn, fp, fn

    @njit
    def recall(tp, fn, round_digit=3):
        if float(tp+fn) > 0:
            return round((tp/float(tp+fn)), round_digit)
        return np.nan

    @njit
    def precision(tp, fp, round_digit=3):
        if tp+fp > 0:
            return round((tp/float(tp+fp)), round_digit)
        return np.nan

    @njit
    def f1(prec, rec, round_digit=3):
        if prec+rec > 0:
            return round((2 * (prec*rec) / float(prec+rec)), round_digit)
        return np.nan

    @njit
    def accuracy(tp, tn, p, n, round_digit=3):
        if p + n > 0:
            return round(((tp + tn) / float(p + n)), round_digit)
        return np.nan

    @njit
    def relative_error_total_energy(predictions, groundtruth, round_digit=3):
        e_pred = np.sum(predictions)
        e_ground = np.sum(groundtruth)
        if float(max(e_pred, e_ground)) > 0:
            return round((np.abs(e_pred - e_ground) / float(max(e_pred, e_ground))), round_digit)
        return np.nan

    @njit
    def mean_absolute_error(predictions, groundtruth, round_digit=3):
        sum_samples = len(predictions)
        if sum_samples > 0:
            total_sum = np.sum(np.abs(predictions - groundtruth))
            return round((total_sum / sum_samples), round_digit)
        return np.nan

    @njit
    def replace_nan_with_0(pr, gr):
        pr[np.isnan(pr)] = 0
        gr[np.isnan(gr)] = 0
        return pr, gr

    @njit
    def thresholding(pr, gr, thres):
        pr = np.array([0 if p < thres else 1 for p in pr])
        gr = np.array([0 if p < thres else 1 for p in gr])
        return pr, gr

    @njit
    def get_positives_negatives(pr):
        pos = np.sum(pr)
        neg = len(pr) - pos
        return pos, neg

    @njit
    def are_equal(pr, gr):
        return np.array_equal(pr, gr)

    if torch.is_tensor(pred):
        pred = pred.numpy()
    else:
        pass

    if torch.is_tensor(ground):
        ground = ground.numpy()
    else:
        pass

    pred, ground = convert_precision_16_to_32(pred), convert_precision_16_to_32(ground)
    print('###### Sanity Check started #######')
    print('Preds == grounds is: ', are_equal(pred, ground))
    print('preds shape {}, ground shape {}'.format(pred.shape, ground.shape))
    print('###### Sanity Check finished ######')
    if ground.shape[0] > 0:
        pred, ground = replace_nan_with_0(pred, ground)
        rete = relative_error_total_energy(pred, ground)
        mae = mean_absolute_error(pred, ground)
        eac = get_eac(pred, ground)
        nde = get_nde(pred, ground)

        pred, ground = thresholding(pred, ground, threshold)

        tp, tn, fp, fn = tp_tn_fp_fn(pred, ground)
        positives, negatives = get_positives_negatives(pred)

        recall = recall(tp, fn, rounding_digit)
        precision = precision(tp, fp, rounding_digit)
        f1 = f1(precision, recall, rounding_digit)
        accuracy = accuracy(tp, tn, positives, negatives, rounding_digit)

        metrics_results = {COLUMN_RECALL: recall, COLUMN_PRECISION: precision,
                           COLUMN_F1: f1, COLUMN_ACCURACY: accuracy,
                           COLUMN_NDE: nde, COLUMN_EAC: eac,
                           COLUMN_MAE: mae, COLUMN_RETE: rete,
                           COLUMN_TP: tp, COLUMN_TN: tn, COLUMN_FP: fp, COLUMN_FN: fn,
                           }
    else:
        print('###### No groundtruth available, metrics are set to NaN ######')
        metrics_results = {COLUMN_RECALL: np.nan, COLUMN_PRECISION: np.nan,
                           COLUMN_F1: np.nan, COLUMN_ACCURACY: np.nan,
                           COLUMN_NDE: np.nan, COLUMN_EAC: np.nan,
                           COLUMN_MAE: np.nan, COLUMN_RETE: np.nan,
                           COLUMN_TP: np.nan, COLUMN_TN: np.nan, COLUMN_FP: np.nan, COLUMN_FN: np.nan,
                           }

    return metrics_results





