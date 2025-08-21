import numpy as np
import pandas as pd
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from multiprocessing import Pool
from tslearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
import os
from tslearn.clustering import TimeSeriesKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.signal import convolve2d

def create_images_for_clustering(detected_swallow_start):
    detected_swallow_start = max(0, detected_swallow_start - 20)
    manometry_swallow = manometry[detected_swallow_start:detected_swallow_start+500, :].T
    
    original_indices = np.arange(manometry_swallow.shape[0])
    new_indices = np.linspace(0, manometry_swallow.shape[0] - 1, manometry_swallow.shape[1])
    interp_func = interp1d(original_indices, manometry_swallow, axis=0, kind='linear', fill_value='extrapolate')
    interpolated_array = np.uint8(interp_func(new_indices))
    orig_image = interpolated_array
    
    orig_image_clustering = cv.resize(interpolated_array, (50, 50))
    orig_image_clustering = cv.GaussianBlur(orig_image_clustering,(9,9),3)

    manometry_swallow = manometry[detected_swallow_start:detected_swallow_start+500, :].T
    grayscale_image_ = cv.GaussianBlur(manometry_swallow.astype(float),(3,3),1)
    kernel = np.array([[-1, 0, 0,0,0, 0, 0, 0,0, 1]])
    conv_image = convolve2d(grayscale_image_, kernel, 'valid')
    pad_ = int((500 - conv_image.shape[1])/2)
    conv_image = np.pad(conv_image, ((0, 0), (pad_, pad_)))
    conv_image = np.uint8(conv_image * conv_image)
    original_indices = np.arange(conv_image.shape[0])
    new_indices = np.linspace(0, conv_image.shape[0] - 1, conv_image.shape[1])
    interp_func = interp1d(original_indices, conv_image, axis=0, kind='linear', fill_value='extrapolate')
    interpolated_array = np.uint8(interp_func(new_indices))
    change_image_clustering = cv.resize(conv_image, (50, 50))
    change_image_clustering = cv.GaussianBlur(change_image_clustering,(9,9),3)

    return orig_image, orig_image_clustering, change_image_clustering

def prepare_for_clustering(img):
    return img.flatten()

def mse(imageA, imageB):
	mse = (np.square(imageA - imageB)).mean()
	return mse

def indices_of_values(array, values):
    return [index for index, element in enumerate(array) if element in values]

# "agglomerative" or "kmeans" or "dtw-kmeans"
cluster_method = "agglomerative"
cluster_using_change_filter = True

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

    with Pool(40) as pool:
        results = pool.map(create_images_for_clustering, predicted_swallows)
    
    orig_images = [j[0] for j in results]
    orig_images_clustering = [j[1] for j in results]
    change_image_clustering = [j[2] for j in results]

    if cluster_using_change_filter:
        imgs_clustering = change_image_clustering
    else:
        imgs_clustering = orig_images_clustering
    
    features = np.array([prepare_for_clustering(img) for img in imgs_clustering])
    
    # Dimensionality reduction - PCA
    pca = PCA(n_components=30)
    features_embedded = pca.fit_transform(features)

    ############################################
    # first clustering
    ############################################

    mean_intra_cluster_distances = []
     # Choose best number of clusters based on lowest intracluster distance
    for num_clusters in range(4, 11):
        if cluster_method == 'kmeans':
            cluster_obj = KMeans(n_clusters=num_clusters).fit(features_embedded)
        elif cluster_method == 'agglomerative':
            cluster_obj = AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=None, compute_distances=True).fit(features_embedded)
        else:
            print("No valid cluster method. Must be kmeans or agglomerative")
        cluster_labels = cluster_obj.labels_

        # Compute intra cluster distances
        pairwise_dist = pairwise_distances(features_embedded)
        intra_cluster_distances = {}
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_distances = pairwise_dist[cluster_indices][:, cluster_indices]
            avg_distance = np.mean(cluster_distances)
            intra_cluster_distances[cluster_id] = avg_distance
    
        distances = []
        for cluster_id, distance in intra_cluster_distances.items():
            distances.append(distance)
        mean_intra_cluster_distance = np.array(distances).mean()
        mean_intra_cluster_distances.append(mean_intra_cluster_distance)

    best_cluster_number = range(4,16)[np.array(mean_intra_cluster_distances).argmin()]

    # Perform clustering with best cluster number as identified above
    num_clusters = best_cluster_number

    if cluster_method == 'kmeans':
        cluster_obj = KMeans(n_clusters=num_clusters).fit(features_embedded)
    elif cluster_method == 'agglomerative':
        cluster_obj = AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=None, compute_distances=True).fit(features_embedded)
    cluster_labels = cluster_obj.labels_
    
    # Compute cluster centers
    cluster_centers = []
    for j in range(num_clusters):
        locations = np.where(cluster_labels == j)
        class_features = features[locations, :][0]
        mean_class = class_features.mean(axis=0)
        cluster_centers.append(mean_class)

    cluster_centers = np.array(cluster_centers)
    cluster_centers = cluster_centers.reshape((-1, 50, 50))

    # Plot cluster histogram
    plt.close()
    fig = plt.figure(figsize=(9, 13))
    ax = ((pd.Series(cluster_labels).value_counts().sort_values()/len(cluster_labels)).round(2).plot(kind='barh'))
    for c in ax.containers:
            ax.bar_label(c)
    plt.tight_layout()
    plt.savefig('clustering_outputs/patient_{}_histogram_{}'.format(patient_id, cluster_method))

    # Identify small clusters with <15% of samples
    cluster_counts = pd.Series(cluster_labels).value_counts()/len(cluster_labels)
    ordered_cluster_index = list(cluster_counts.index)
    small_clusters = list(cluster_counts[cluster_counts < 0.15].index)

    # Compute distance to respective cluster center for each sample
    distances_to_center = [mse(img, cluster_centers[cluster_labels[i]].flatten()) for i, img in enumerate(features)]
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
    closest_images = [orig_images[path] if path != -1 else np.ones(orig_images[0].shape, dtype=np.uint8) for path in closest_samples]

    furthest_samples = []
    for i in ordered_cluster_index:
        x = dist_df[dist_df['cluster_nr'] == i]['distance'][-nr_closest_samples:]
        x = list(x.index)
        for m in range(nr_closest_samples - len(x)):
            x.append(-1)
        furthest_samples.extend(x)  
    furthest_images = [orig_images[path] if path != -1 else np.ones(orig_images[0].shape, dtype=np.uint8) for path in furthest_samples]

    # Plot closest and furthest samples of each cluster center
    plt.close()
    fig = plt.figure(figsize=(9, 13))
    columns = nr_closest_samples + 1
    rows = num_clusters  
    i = 0
    j = 0
    while j < columns*rows:
        if j%(nr_closest_samples + 1)==0:
            img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :]
            fig.add_subplot(rows, columns, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            j+=1
        img = closest_images[i]
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
    plt.savefig('clustering_outputs/patient_{}_closest_samples_{}'.format(patient_id, cluster_method))

    plt.close()
    fig = plt.figure(figsize=(9, 13))
    columns = nr_closest_samples + 1
    rows = num_clusters  
    i = 0
    j = 0
    while j < columns*rows:
        if j%(nr_closest_samples + 1)==0:
            img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :]
            fig.add_subplot(rows, columns, j+1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            j+=1
        img = furthest_images[i]
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
    plt.savefig('clustering_outputs/patient_{}_furthest_samples_{}'.format(patient_id, cluster_method))
    
    ############################################
    # second clustering of small clusters
    ############################################
    if len(small_clusters) > 0:

        small_cluster_indices = indices_of_values(cluster_labels, small_clusters)
        
        features_small_cluster = features[small_cluster_indices]
        features_embedded_small_cluster = features_embedded[small_cluster_indices]
        orig_images_small_cluster = list(np.array(orig_images)[small_cluster_indices])

        # Number of clusters
        num_clusters = 10

        if  cluster_method == 'kmeans':
            cluster_obj = KMeans(n_clusters=num_clusters).fit(features_embedded_small_cluster)
        else:
            cluster_obj = AgglomerativeClustering(n_clusters=num_clusters, distance_threshold=None, compute_distances=True).fit(features_embedded_small_cluster)
        cluster_labels = cluster_obj.labels_

        # Calculate cluster centers
        cluster_centers = []
        for j in range(num_clusters):
            locations = np.where(cluster_labels == j)
            class_features = np.array(features_small_cluster)[locations, :][0]
            mean_class = class_features.mean(axis=0)
            cluster_centers.append(mean_class)
        cluster_centers = np.array(cluster_centers)
        cluster_centers = cluster_centers.reshape((-1, 50, 50))

        # Plot cluster histogram
        plt.close()
        fig = plt.figure(figsize=(9, 13))
        ax = ((pd.Series(cluster_labels).value_counts().sort_values()/len(predicted_swallows)).round(2).plot(kind='barh'))
        for c in ax.containers:
            ax.bar_label(c)
        plt.tight_layout()
        plt.savefig('clustering_outputs/patient_{}_special_clusters_histogram_{}'.format(patient_id, cluster_method))

        cluster_counts = pd.Series(cluster_labels).value_counts()/len(cluster_labels)
        ordered_cluster_index = list(cluster_counts.index)            
        
        # Compute distance to respective cluster center for each sample
        distances_to_center = [mse(img, cluster_centers[cluster_labels[i]].reshape((-1,))) for i, img in enumerate(features_small_cluster)]
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
        closest_images = [orig_images_small_cluster[path] if path != -1 else np.ones(orig_images_small_cluster[0].shape, dtype=np.uint8) for path in closest_samples]

        furthest_samples = []
        for i in ordered_cluster_index:
            x = dist_df[dist_df['cluster_nr'] == i]['distance'][-nr_closest_samples:]
            x = list(x.index)
            for m in range(nr_closest_samples - len(x)):
                x.append(-1)
            furthest_samples.extend(x)
        furthest_images = [orig_images_small_cluster[path] if path != -1 else np.ones(orig_images_small_cluster[0].shape, dtype=np.uint8) for path in furthest_samples]
 
        # Plot closest and furthest samples of each cluster center
        plt.close()
        fig = plt.figure(figsize=(9, 13))
        columns = nr_closest_samples + 1
        rows = num_clusters  
        i = 0
        j = 0
        while j < columns*rows:
            if j%(nr_closest_samples + 1)==0:
                img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :]
                fig.add_subplot(rows, columns, j+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)
                j+=1
            img = closest_images[i]
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
        plt.savefig('clustering_outputs/patient_{}_special_clusters_closest_samples_{}'.format(patient_id, cluster_method))

        plt.close()
        fig = plt.figure(figsize=(9, 13))
        columns = nr_closest_samples + 1
        rows = num_clusters  
        i = 0
        j = 0
        while j < columns*rows:
            if j%(nr_closest_samples + 1)==0:
                img = cluster_centers[ordered_cluster_index[int(j/columns)], :, :]
                fig.add_subplot(rows, columns, j+1)
                plt.xticks([])
                plt.yticks([])
                plt.imshow(img)
                j+=1
            img = furthest_images[i]
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
        plt.savefig('clustering_outputs/patient_{}_special_clusters_furthest_samples_{}'.format(patient_id, cluster_method))