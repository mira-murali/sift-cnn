import numpy as np
import cv2
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL
import pickle
import sklearn.cluster

sift = cv2.xfeatures2d.SIFT_create()
surf = cv2.xfeatures2d.SURF_create(300)

def compute_sift(image):
    # Expects PIL image
    image = np.array(image)
    image = image[:, :, ::-1]
    __, sift_desc = sift.detectAndCompute(image, None)
    return sift_desc

def compute_surf(image):
    # Expects PIL image
    image = np.array(image)
    image = image[:, :, ::-1]
    __, surf_desc = surf.detectAndCompute(image, None)
    return surf_desc

def compute_KMeans(feature_vectors, num_clusters = 128):
    kmeans = sklearn.cluster.KMeans(n_clusters=num_clusters, n_jobs=6).fit(feature_vectors)
    return kmeans

def get_sift_features(data):
    features = []
    for i in range(len(data)):
        image, _ = data.__getitem__(i)
        sift_feature = compute_sift(image)
        if sift_feature is None:
            sift_feature = np.zeros((1, 128))
        features.append(sift_feature)
    return np.vstack(features)

def get_surf_features(data):
    features = []
    for i in range(len(data)):
        image, _ = data.__getitem__(i)
        surf_feature = compute_surf(image)
        if surf_feature is None:
            surf_feature = np.zeros((1, 64))
        features.append(surf_feature)
    return np.vstack(features)

def get_histogram_sift(image, kmeans):
    image = np.array(image)
    image = image[:, :, ::-1]
    __, sift_desc = sift.detectAndCompute(image, None)
    if sift_desc is None:
        sift_desc = np.zeros((1, 128))
    labels = kmeans.predict(sift_desc)
    return np.histogram(labels, bins=64, range=(0.0, 64.0))

def get_histogram_surf(image, kmeans):
    image = np.array(image)
    image = image[:, :, ::-1]
    __, surf_desc = surf.detectAndCompute(image, None)
    if surf_desc is None:
        surf_desc = np.zeros((1, 64))
    labels = kmeans.predict(surf_desc)
    return np.histogram(labels, bins=64, range=(0.0, 64.0))
    

if __name__ == '__main__':
    data = datasets.CIFAR10('../cifar10', train=True, transform=None, download=True)
    print('Got data.')
    print('Getting sift features.')
    sift_feature_vectors = get_sift_features(data)
    print('Got sift features.')
    print('Getting surf features.')
    surf_feature_vectors = get_surf_features(data)
    print('Got surf features.')
    print('Doing SIFT KMeans.')
    sift_kmeans = compute_KMeans(sift_feature_vectors)
    print('SIFT KMeans done.')
    print('Doing SURF KMeans.')
    surf_kmeans = compute_KMeans(surf_feature_vectors)
    print('SURF KMeans done.')    
    pickle.dump(sift_kmeans, open('sift_kmeans_obj_64', 'wb'))
    pickle.dump(surf_kmeans, open('surf_kmeans_obj_64', 'wb'))
    print('KMeans dumped.')