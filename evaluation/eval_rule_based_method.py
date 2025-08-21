import numpy as np
import pandas as pd
from multiprocessing import Pool
from scipy.signal import find_peaks

d = 400

def test_recall(labels):
    found = False
    dist=None
    for found_swallow in found_swallows:
        likely_start = found_swallow
        if likely_start in list(range(swallow.iloc[labels]['start'], swallow.iloc[labels]['start']+d)):
            correct = 1
            found=True
            dist=likely_start-swallow.iloc[labels]['start']
            break
    if not found:
        correct = 0
    return correct, dist

def test_precision(found_swallow):
    found = False
    dist=None
    likely_start = found_swallow
    for swallow_i in swallow['start']:
        if likely_start in list(range(swallow_i, swallow_i+d)):
            correct = 1
            found=True
            dist=likely_start-swallow_i
            break
    if not found:
        correct = 0
    return correct, dist

recalls = []
precisions = []
f1s = []

for patient_nr in range(1, 26):

    # load manometry file
    data_file = '../data/manometry_{}.csv'.format(patient_nr)
    manometry = pd.read_csv(data_file, sep=',')
    # load swallow labels
    swallow_file = '../data/manometry_swallows_{}.csv'.format(patient_nr)
    swallow = pd.read_csv(swallow_file, sep=',')

    # preprocess manometry values
    window_size = 30
    manometry = manometry.rolling(window=window_size, center=True).mean()
    manometry = np.array(manometry.fillna(0))
    manometry = manometry.clip(-200, 300)
    manometry = (255*(manometry - np.min(manometry))/np.ptp(manometry)).astype(int)
    all_data = pd.DataFrame(manometry).fillna(0)

    # Binary mask, 1 where value > 80
    all_data = pd.DataFrame(np.where(all_data > 80, 1, 0), index=all_data.index, columns=all_data.columns)
    # Apply running sum
    window_size = 20
    all_data = all_data.rolling(window=window_size, center=True).sum().fillna(0)
    # Sum over sensors
    sum_per_timestep = all_data.sum(axis=1)
    sum_per_timestep_rolling = pd.Series(sum_per_timestep).rolling(window=100, center=True).mean().fillna(0)

    # Find peaks in vector
    found_swallows, properties = find_peaks(sum_per_timestep_rolling, distance=400, prominence=(5, None), height=20)

    # Calculate metrics
    with Pool(20) as pool:
        results_recall = pool.map(test_recall, range(len(swallow)))
    found_list_recall = [j[0] for j in results_recall]
    dist_list_recall = [j[1] for j in results_recall]
    
    with Pool(20) as pool:
        results_precision = pool.map(test_precision, found_swallows)
    found_list_precision = [j[0] for j in results_precision]
    dist_list_precision = [j[1] for j in results_precision]
    
    recall = np.array(found_list_recall).sum()/len(swallow)
    precision = np.array(found_list_precision).sum()/len(found_swallows)
    f1 = (2 * recall * precision)/(recall + precision)

    recalls.append(recall)
    precisions.append(precision)
    f1s.append(f1)
    
    # save predictions
    labels_df = pd.DataFrame({'found_start': found_swallows})
    labels_df.to_csv('rule_based_predictions/pred_patient_{}.csv'.format(patient_nr), index=False)

# save metrics
labels_df = pd.DataFrame({'recall': recalls, 'precision': precisions, 'f1': f1s})
labels_df.to_csv('rule_based_predictions/scores.csv', index=False)

