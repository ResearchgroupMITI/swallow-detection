import numpy as np
import pandas as pd
from multiprocessing import Pool

d = 400

def threshold_to_binary(arr, threshold):
    return (arr >= threshold).astype(int)

def find_groups_of_ones(binary_array, min_distance):
    groups = []
    start = None
    for i, bit in enumerate(binary_array):
        if bit == 1:
            if start is None:
                start = i
        else:
            if start is not None:
                groups.append((start, i - 1))
                start = None

    # Check if there's a group ending at the last index
    if start is not None:
        groups.append((start, len(binary_array) - 1))

    # Merge adjacent groups that are below the minimum distance
    merged_groups = []
    if groups:
        merged_groups.append(groups[0])
        for group in groups[1:]:
            if group[0] - merged_groups[-1][1] <= min_distance:
                merged_groups[-1] = (merged_groups[-1][0], group[1])
            else:
                merged_groups.append(group)

    return merged_groups

def test_recall(labels):
    found = False
    dist=None
    for found_swallow in found_swallows:
        likely_start = found_swallow[0] + swallow_conf[found_swallow[0]:found_swallow[1]].argmax()
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
    likely_start = found_swallow[0] + swallow_conf[found_swallow[0]:found_swallow[1]].argmax()
    for swallow_i in swallow['start']:
        if likely_start in list(range(swallow_i-int(d/2), swallow_i+int(d/2))):
            correct = 1
            found=True
            dist=likely_start-swallow_i
            break
    if not found:
        correct = 0
    return correct, dist

patients = [[5,7,10,20,21], [12, 19, 22, 23, 25], [3, 6, 11, 15, 17], [2, 9, 13, 14, 18], [1, 4, 8, 16, 24]]

recalls = []
precisions = []
f1s = []

for fold in range(5):
    for patient_nr in patients[fold]:
        fold_nr = fold+1

        # load model inference output
        df = pd.read_pickle("../output_MobileNet/output_fold{}_patient{}.pkl".format(fold_nr, patient_nr))

        # load swallow labels
        swallow_file = '../data/manometry_swallows_{}.csv'.format(patient_nr)
        swallow = pd.read_csv(swallow_file, sep=',')
        
        indices = df['indices']
        probabilities = df['probability']
        outputs = df['output']
        # confidence of swallows
        swallow_conf = np.array(outputs) * np.array(probabilities)
        swallow_conf_smoothed = np.array(pd.Series(swallow_conf).rolling(window=20, center=True).mean().fillna(0))

        # only keep confidences above defined threshold
        threshold = 0.2
        swallow_conf_certain = threshold_to_binary(swallow_conf_smoothed, threshold)

        # find groups of ones with minimum distance between each other
        min_distance = 200
        found_swallows = find_groups_of_ones(swallow_conf_certain, min_distance)

        # groups of ones must have a minimum length
        min_length = 2
        found_swallows = [(start,end) for (start,end) in found_swallows if end-start > min_length]

        # Calculate metrics
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

        # save predictions        
        detected_swallow_starts = []
        for found_swallow in found_swallows:
                detected_swallow_starts.append(found_swallow[0] + swallow_conf_smoothed[found_swallow[0]:found_swallow[1]].argmax())
        pd.DataFrame({'detected_swallow_starts':  pd.Series(detected_swallow_starts), 'true_swallow_starts': swallow['start'], 'found_true': pd.Series(found_list_recall)}).to_csv('ml_predictions/patient_{}.csv'.format(patient_nr), index=False)

# save metrics
labels_df = pd.DataFrame({'recall': recalls, 'precision': precisions, 'f1': f1s})
labels_df.to_csv('ml_predictions/scores.csv', index=False)
