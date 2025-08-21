import numpy as np
import pandas as pd
from multiprocessing import Pool

d = 400

def test_recall(labels):
    found = False
    dist=None
    for found_swallow in found_swallows:
        likely_start = found_swallow
        if likely_start in list(range(swallow.iloc[labels]['start']-int(d/2), swallow.iloc[labels]['start']+int(d/2))):
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
        if likely_start in list(range(swallow_i-int(d/2), swallow_i+int(d/2))):
            correct = 1
            found=True
            dist=likely_start-swallow_i
            break
    if not found:
        correct = 0
    return correct, dist

# patients where Vimedat prediction is available
patients = [5,7,20,21,12,19,22,25,3,6,11,15,17,9,13,14,18,1,4,8,16]

recalls = []
precisions = []
f1s = []

for patient_nr in patients:

    # load swallow labels
    swallow_file = '../data/manometry_swallows_{}.csv'.format(patient_nr)
    swallow = pd.read_csv(swallow_file, sep=',')

    # load vimedat predictions
    detection_file = '../vimedat_predictions/patient_{}.csv'.format(patient_nr)
    detected_swallows = pd.read_csv(detection_file, sep=',')
    found_swallows  = list(detected_swallows['start'])

    with Pool(20) as pool:
        results_recall = pool.map(test_recall, range(len(swallow)))
    found_list_recall = [j[0] for j in results_recall]
    dist_list_recall = [j[1] for j in results_recall]

    with Pool(20) as pool:
        results_precision = pool.map(test_precision, found_swallows)
    found_list_precision = [j[0] for j in results_precision]
    dist_list_precision = [j[1] for j in results_precision]

    print('Total swallows:', len(swallow), ' Found swallows:', np.array(found_list_recall).sum(), 'Found rate:', np.array(found_list_recall).sum()/len(found_list_recall))

    print('Total predicted swallows:', len(found_swallows), ' True swallows:', np.array(found_list_precision).sum(), 'True prediction rate:', np.array(found_list_precision).sum()/len(found_swallows))

    recall = np.array(found_list_recall).sum()/len(swallow)
    precision = np.array(found_list_precision).sum()/len(found_swallows)
    f1 = (2 * recall * precision)/(recall + precision)

    recalls.append(recall)
    precisions.append(precision)
    f1s.append(f1)

    detected_swallow_starts = []
    for found_swallow in found_swallows:
            detected_swallow_starts.append(found_swallow)

    pd.DataFrame({'detected_swallow_starts':  pd.Series(detected_swallow_starts), 'true_swallow_starts': swallow['start'], 'found_true': pd.Series(found_list_recall)}).to_csv('vimedat_predictions/final_predictions/patient_{}.csv'.format(patient_nr), index=False)

labels_df = pd.DataFrame({'recall': recalls, 'precision': precisions, 'f1': f1s})
labels_df.to_csv('vimedat_predictions/scores.csv', index=False)
