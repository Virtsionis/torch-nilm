import math
import numpy as np

def metrics(pred, ground, threshold=40, mmax=None, round_digit=3):

    def tp_tn_fp_fn(states_pred, states_ground):
        tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
        fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
        fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
        tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
        return tp, tn, fp, fn

    def recall(tp,fn):
        return tp/float(tp+fn)

    def precision(tp,fp):
        return tp/float(tp+fp)

    def f1(prec,rec):
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

    pred[np.isnan(pred)] = 0
    ground[np.isnan(ground)] = 0

    RETE = round(relative_error_total_energy(pred, ground),round_digit)
    MAE = mean_absolute_error(pred, ground)

    if mmax:
        MAE *= mmax
    MAE = round(MAE,round_digit)

    pred = np.array([0 if (p)<threshold else 1 for p in pred])
    ground = np.array([0 if p<threshold else 1 for p in ground])

    tp, tn, fp, fn = tp_tn_fp_fn(pred,ground)
    positives = sum(pred)
    negatives = len(pred) - positives

    recall = round(recall(tp,fn),round_digit)
    precision = round(precision(tp,fp),round_digit)
    f1 = round(f1(precision,recall),round_digit)
    accuracy = round(accuracy(tp,tn,positives,negatives),round_digit)

    metrics_results ={"recall": recall, "precision": precision,
                      "f1": f1, "accuracy": accuracy,
                      "MAE": MAE, "RETE": RETE}
    return metrics_results