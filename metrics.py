from nilmtk.electric import align_two_meters
import numpy as np
import math
import csv

def tp_tn_fp_fn(states_pred, states_ground):
    tp = np.sum(np.logical_and(states_pred == 1, states_ground == 1))
    fp = np.sum(np.logical_and(states_pred == 1, states_ground == 0))
    fn = np.sum(np.logical_and(states_pred == 0, states_ground == 1))
    tn = np.sum(np.logical_and(states_pred == 0, states_ground == 0))
    return tp, tn, fp, fn

def recall_precision_accuracy_f1(pred, ground, pr_threshold = None, gr_threshold = None):
    aligned_meters = align_two_meters(pred, ground)
    if pr_threshold == None:
        pr_threshold = ground.on_power_threshold() #If not TH was provided, both sets get TH from ground
    if gr_threshold == None:
        gr_threshold = ground.on_power_threshold()  # If not TH was provided, both sets get TH from ground
    print('True threshold: ',gr_threshold)
    print('Pred threshold: ', pr_threshold)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        sum_samples += len(chunk)
        pr = chunk.iloc[:,0].fillna(0) #method='bfill'
        gr = chunk.iloc[:,1].fillna(0)
        pr = np.array([0 if (p)<pr_threshold else 1 for p in pr])
        gr = np.array([0 if p<gr_threshold else 1 for p in gr])

        tp, tn, fp, fn = tp_tn_fp_fn(pr,gr)
        p = sum(pr)
        n = len(pr) - p

        chunk_results.append([tp,tn,fp,fn,p,n])

    if sum_samples == 0:
        return None
    else:
        [tp,tn,fp,fn,p,n] = np.sum(chunk_results, axis=0)

        res_recall = recall(tp,fn)
        res_precision = precision(tp,fp)
        res_f1 = f1(res_precision,res_recall)
        res_accuracy = accuracy(tp,tn,p,n)

        return (res_recall,res_precision,res_accuracy,res_f1)

def relative_error_total_energy(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    chunk_results = []
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        E_pred = sum(chunk.iloc[:,0])
        E_ground = sum(chunk.iloc[:,1])

        chunk_results.append([
                            E_pred,
                            E_ground
                            ])
    if sum_samples == 0:
        return None
    else:
        [E_pred, E_ground] = np.sum(chunk_results,axis=0)
        return abs(E_pred - E_ground) / float(max(E_pred,E_ground))

def mean_absolute_error(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        total_sum += sum(abs((chunk.iloc[:,0]) - chunk.iloc[:,1]))
    if sum_samples == 0:
        return None
    else:
        return total_sum / sum_samples


def recall(tp,fn):
    return tp/float(tp+fn)

def precision(tp,fp):
    return tp/float(tp+fp)

def f1(prec,rec):
    return 2 * (prec*rec) / float(prec+rec)

def accuracy(tp, tn, p, n):
    return (tp + tn) / float(p + n)

# ====================================
# Metrics bellow here have not been tested much, so they might not be completely right
# ====================================

def RMSE(pred, ground):
    aligned_meters = align_two_meters(pred, ground)
    total_sum = 0.0
    sum_samples = 0.0
    for chunk in aligned_meters:
        chunk.fillna(0, inplace=True)
        sum_samples += len(chunk)
        total_sum += sum(np.power((chunk.iloc[:,0]) - chunk.iloc[:,1],2))
    if sum_samples == 0:
        return None
    else:
        return math.sqrt(total_sum / sum_samples)

def TECA(pred_list, ground_list, mains):
    #pred_list and ground_list will contain the meters for each appliace
    listSize = len(pred_list)
    total_diff_sum = 0.0
    total_aggr_sum = 0.0
    sum_samples = 0.0
    for i in range(listSize):
        pred = pred_list[i]
        ground = ground_list[i]
        aligned_meters_pg = align_two_meters(pred, ground)
        for chunk in aligned_meters_pg:
            chunk.fillna(0, inplace=True)
            sum_samples += len(chunk)
            total_diff_sum += sum(abs((chunk.iloc[:, 0]) - chunk.iloc[:, 1]))
            if(i==0): #count the total timestamps
                sum_samples += len(chunk)

    #Aggregate sum
    aligned_meters_pm = align_two_meters(pred, mains)
    for chunk_mains in aligned_meters_pm:
        chunk_mains.fillna(0, inplace=True)
        total_aggr_sum += sum(chunk_mains.iloc[:, 1])

    if sum_samples == 0:
        return None
    else:
        return 1 - (1/2)*(total_diff_sum/total_aggr_sum)

def writeResultsToCSV(resultsDict, outFileName, clearFile):
    # --- if clearFile is true, clear file before writing. Else append. SEARCH how to achieve this ---

    if(clearFile):
        with open(outFileName, 'w') as csvfile:
            fieldnames = resultsDict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow(resultsDict)
    else:
        with open(outFileName, 'a') as csvfile:
            fieldnames = resultsDict.keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow(resultsDict)

def GeneralizationOverUnseenHouses_fromCSV(outFileName, generalizatonMetric):
    #The output file is supposed to have only the data needed for this metric. I.e.
    #For ONE model, each house and each of it's meters (or at least not have data from multiple models-algorithms)
    with open(outFileName) as csvfile:
        reader = csv.DictReader(csvfile)
        totalAccSum = 0.0
        totalExperiments = 0.0
        totalAccPerHouse = {}
        totalExpPerHouse = {}
        for row in reader:
            building = row['building']
            totalAccSum += float(row[generalizatonMetric]) #Update total
            totalExperiments += 1.0

            curHouseAcc = totalAccPerHouse.get(building,0.0)
            curHouseExps = totalExpPerHouse.get(building,0.0)
            totalAccPerHouse[building] = curHouseAcc + float(row[generalizatonMetric])
            totalExpPerHouse[building] = curHouseExps + 1.0

        if totalExperiments == 0:
            return None

        totalAvgAcc = totalAccSum / totalExperiments
        GoUH_Sum = 0
        for building in totalAccPerHouse.keys():
            buildingAvgAcc = totalAccPerHouse[building] / totalExpPerHouse[building]
            GoUH_Sum += pow(buildingAvgAcc - totalAvgAcc,2)

        return math.sqrt(GoUH_Sum/totalExperiments)