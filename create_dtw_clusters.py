import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from multiprocessing import Pool
from tslearn import metrics
import os
from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.signal import convolve2d

def prepare_for_clustering(img):
    return img.flatten()

def mse(imageA, imageB):
	mse = (np.square(imageA - imageB)).mean()
	return mse

def indices_of_values(array, values):
    return [index for index, element in enumerate(array) if element in values]

def create_ts_for_clustering(swallow_time_series):
    blurred_ts = cv.GaussianBlur(swallow_time_series.astype(float),(3,3),1)
    blurred_ts = cv.resize(blurred_ts, (36, 500))
    
    normal_ts = cv.resize(blurred_ts, (50, 50))

    kernel = np.array([[-1, 0, 0,0,0, 0, 0, 0,0, 1]])
    conv_ts = convolve2d(blurred_ts.copy().T, kernel, 'valid')
    pad_ = int((500 - conv_ts.shape[1])/2)
    conv_ts = np.pad(conv_ts, ((0, 0), (pad_, pad_)))

    conv_ts = np.uint8(conv_ts * conv_ts)
    original_indices = np.arange(conv_ts.shape[0])
    new_indices = np.linspace(0, conv_ts.shape[0] - 1, conv_ts.shape[1])
    interp_func = interp1d(original_indices, conv_ts, axis=0, kind='linear', fill_value='extrapolate')
    conv_ts = np.uint8(interp_func(new_indices))
    conv_ts = cv.resize(conv_ts, (50, 50))
    conv_ts = cv.GaussianBlur(conv_ts,(9,9),3)

    return normal_ts, conv_ts, blurred_ts

for patient_id in range(1,26):

    data_file = 'data/manometry_{}.csv'.format(patient_id)
    manometry = pd.read_csv(data_file, sep=',')
    swallow_file = 'data/manometry_swallows_{}.csv'.format(patient_id)
    swallow = pd.read_csv(swallow_file, sep=',')
    predicted_swallows = list(pd.read_csv('inference_outputs/final_predictions_mobilenet/patient_{}.csv'.format(patient_id), sep=',')['detected_swallow_starts'].dropna().astype(int))

    window_size = 30
    manometry = manometry.rolling(window=window_size).mean()
    manometry = np.array(manometry.dropna())
    manometry = manometry.clip(-200, 300)
    manometry = (255*(manometry - np.min(manometry))/np.ptp(manometry)).astype(int)

    swallow_time_series_list = []
    img_nr = 0
    for i in range(len(predicted_swallows)):
        swallow_start = int(predicted_swallows[i])
        swallow_end = swallow_start + 500
        
        manometry_swallow = manometry[swallow_start:swallow_end, :]
        swallow_time_series_list.append(manometry_swallow)

    with Pool(40) as pool:
        results = pool.map(create_ts_for_clustering, swallow_time_series_list)
    
    normal_ts_list = [j[0] for j in results]
    change_ts_list = [j[1].T for j in results]
    orig_ts_list = [j[2] for j in results]

    features_change_ts = np.array(change_ts_list)
    features_normal_ts = np.array(normal_ts_list)
    features_orig_ts = orig_ts_list

    num_clusters = 10

    model = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", max_iter=10, n_init=1, verbose=0, n_jobs=20)
    model.fit(features_change_ts)

    cluster_labels = model.labels_

    cluster_centers = []
    for j in range(num_clusters):
        locations = np.where(cluster_labels == j)
        class_features = features_change_ts[locations, :, :][0]
        mean_class = class_features.mean(axis=0)
        cluster_centers.append(mean_class)
    cluster_centers = np.array(cluster_centers)


    # Plot cluster histogram
    plt.close()
    fig = plt.figure(figsize=(9, 13))
    ax = ((pd.Series(cluster_labels).value_counts().sort_values()/len(cluster_labels)).round(2).plot(kind='barh'))
    for c in ax.containers:
            ax.bar_label(c)
    plt.tight_layout()
    plt.savefig('clustering_outputs/patient_{}_histogram_{}'.format(patient_id, "DTW-k-means"))

    # Identify small clusters with <15% of samples
    cluster_counts = pd.Series(cluster_labels).value_counts()/len(cluster_labels)
    ordered_cluster_index = list(cluster_counts.index)
    small_clusters = list(cluster_counts[cluster_counts < 0.15].index)

    # Compute distance to respective cluster center for each sample
    distances_to_center = [mse(img.flatten(), cluster_centers[cluster_labels[i]].flatten()) for i, img in enumerate(features_change_ts)]
    dist_df = pd.DataFrame({'distance': distances_to_center, "cluster_nr": cluster_labels})
    dist_df = dist_df.sort_values('distance')

    # Get closest and furthest samples of each cluster center
    nr_closest_samples = 5

    closest_samples = []
    for i in ordered_cluster_index:
        x = dist_df[dist_df['cluster_nr'] == i]['distance'][:nr_closest_samples]
        x = list(x.index)
        for m in range(nr_closest_samples - len(x)):
            x.append(-1)
        closest_samples.extend(x)
    closest_images = [features_orig_ts[path] if path != -1 else np.ones(features_orig_ts[0].shape, dtype=np.uint8) for path in closest_samples]

    furthest_samples = []
    for i in ordered_cluster_index:
        x = dist_df[dist_df['cluster_nr'] == i]['distance'][-nr_closest_samples:]
        x = list(x.index)
        for m in range(nr_closest_samples - len(x)):
            x.append(-1)
        furthest_samples.extend(x)  
    furthest_images = [features_orig_ts[path] if path != -1 else np.ones(features_orig_ts[0].shape, dtype=np.uint8) for path in furthest_samples]

    # Plot closest and furthest samples of each cluster center
    plt.close()
    fig = plt.figure(figsize=(9, 13))
    columns = nr_closest_samples + 1
    rows = num_clusters  
    i = 0
    j = 0
    while j < columns*rows:
        if j%(nr_closest_samples + 1)==0:
            img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :].T
            fig.add_subplot(rows, columns, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            j+=1
        img = closest_images[i].T
        img = cv.resize(img.astype(np.uint8), (250, 250))
        img = cv.applyColorMap(img, cv.COLORMAP_JET)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, j+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        j+=1
        i+=1
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('clustering_outputs/patient_{}_closest_samples_{}'.format(patient_id, "DTW-k-kmeans"))

    plt.close()
    fig = plt.figure(figsize=(9, 13))
    columns = nr_closest_samples + 1
    rows = num_clusters  
    i = 0
    j = 0
    while j < columns*rows:
        if j%(nr_closest_samples + 1)==0:
            img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :].T
            fig.add_subplot(rows, columns, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            j+=1
        img = furthest_images[i].T
        img = cv.resize(img.astype(np.uint8), (250, 250))
        img = cv.applyColorMap(img, cv.COLORMAP_JET)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        fig.add_subplot(rows, columns, j+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        j+=1
        i+=1
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('clustering_outputs/patient_{}_furthest_samples_{}'.format(patient_id, "DTW-k-kmeans"))
    
    ############################################
    # second clustering of small clusters
    ############################################
    if len(small_clusters) > 0:

        small_cluster_indices = indices_of_values(cluster_labels, small_clusters)
        
        features_change_ts_small = features_change_ts[small_cluster_indices]
        features_normal_ts_small = features_normal_ts[small_cluster_indices]
        features_orig_ts_small = list(np.array(features_orig_ts)[small_cluster_indices])

        # Number of clusters
        num_clusters = 10

        model = TimeSeriesKMeans(n_clusters=num_clusters, metric="dtw", max_iter=10, n_init=1, verbose=0, n_jobs=20)
        model.fit(features_change_ts_small)
    
        cluster_labels = model.labels_

        cluster_centers = []
        for j in range(num_clusters):
            locations = np.where(cluster_labels == j)
            class_features = features_change_ts_small[locations, :, :][0]
            mean_class = class_features.mean(axis=0)
            cluster_centers.append(mean_class)
        cluster_centers = np.array(cluster_centers)

       # Plot cluster histogram
        plt.close()
        fig = plt.figure(figsize=(9, 13))
        ax = ((pd.Series(cluster_labels).value_counts().sort_values()/len(cluster_labels)).round(2).plot(kind='barh'))
        for c in ax.containers:
                ax.bar_label(c)
        plt.tight_layout()
        plt.savefig('clustering_outputs/patient_{}_special_clusters_histogram_{}'.format(patient_id, "DTW-k-means"))

          # Identify small clusters with <15% of samples
        cluster_counts = pd.Series(cluster_labels).value_counts()/len(cluster_labels)
        ordered_cluster_index = list(cluster_counts.index)
    
        # Compute distance to respective cluster center for each sample
        distances_to_center = [mse(img.flatten(), cluster_centers[cluster_labels[i]].flatten()) for i, img in enumerate(features_change_ts_small)]
        dist_df = pd.DataFrame({'distance': distances_to_center, "cluster_nr": cluster_labels})
        dist_df = dist_df.sort_values('distance')

        # Get closest and furthest samples of each cluster center
        nr_closest_samples = 5
    
        closest_samples = []
        for i in ordered_cluster_index:
            x = dist_df[dist_df['cluster_nr'] == i]['distance'][:nr_closest_samples]
            x = list(x.index)
            for m in range(nr_closest_samples - len(x)):
                x.append(-1)
            closest_samples.extend(x)
        closest_images = [features_orig_ts_small[path] if path != -1 else np.ones(features_orig_ts_small[0].shape, dtype=np.uint8) for path in closest_samples]
    
        furthest_samples = []
        for i in ordered_cluster_index:
            x = dist_df[dist_df['cluster_nr'] == i]['distance'][-nr_closest_samples:]
            x = list(x.index)
            for m in range(nr_closest_samples - len(x)):
                x.append(-1)
            furthest_samples.extend(x)  
        furthest_images = [features_orig_ts_small[path] if path != -1 else np.ones(features_orig_ts_small[0].shape, dtype=np.uint8) for path in furthest_samples]

        # Plot closest and furthest samples of each cluster center
        plt.close()
        fig = plt.figure(figsize=(9, 13))
        columns = nr_closest_samples + 1
        rows = num_clusters  
        i = 0
        j = 0
        while j < columns*rows:
            if j%(nr_closest_samples + 1)==0:
                img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :].T
                fig.add_subplot(rows, columns, j+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)
                j+=1
            img = closest_images[i].T
            img = cv.resize(img.astype(np.uint8), (250, 250))
            img = cv.applyColorMap(img, cv.COLORMAP_JET)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            fig.add_subplot(rows, columns, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            j+=1
            i+=1
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig('clustering_outputs/patient_{}_special_clusters_closest_samples_{}'.format(patient_id, "DTW-k-kmeans"))
    
        plt.close()
        fig = plt.figure(figsize=(9, 13))
        columns = nr_closest_samples + 1
        rows = num_clusters  
        i = 0
        j = 0
        while j < columns*rows:
            if j%(nr_closest_samples + 1)==0:
                img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :].T
                fig.add_subplot(rows, columns, j+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)
                j+=1
            img = furthest_images[i].T
            img = cv.resize(img.astype(np.uint8), (250, 250))
            img = cv.applyColorMap(img, cv.COLORMAP_JET)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            fig.add_subplot(rows, columns, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            j+=1
            i+=1
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig('clustering_outputs/patient_{}_special_clusters_furthest_samples_{}'.format(patient_id, "DTW-k-kmeans"))