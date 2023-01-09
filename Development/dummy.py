import os, codecs
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image


# get file
def get_file_name(path):
    files = []
# r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if '.jpeg' in file:
                files.append(os.path.join(r, file))
    return files

def kmeans_detect(file_list, cluster_nums, randomState=None):
    features = []
    files = file_list
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()
    for file in files:
        img = cv2.imread(file)
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp,des = sift.detectAndCompute(gray, None)

        if des is None:
            file_list.remove(file)
            continue
        reshape_feature = des.reshape(-1, 1)
        features.append(reshape_feature[0].tolist())

    input_x = np.array(features)
    kmeans = KMeans(n_clusters=cluster_nums, random_state=randomState).fit(input_x)
    return kmeans.labels_, kmeans.cluster_centers_

def kmeans_detect_single(img, cluster_nums, randomState=None):
    features = []
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    sift = cv2.SIFT_create()

    img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp, des = sift.detectAndCompute(gray, None)

    reshape_feature = des.reshape(-1, 1)
    features.append(reshape_feature[0].tolist())

    input_x = np.array(features)
    kmeans = KMeans(n_clusters=cluster_nums, random_state=randomState).fit(input_x)
    return kmeans.labels_, kmeans.cluster_centers_

## train dummy model
def train_dummy(path, cluster_nums):
    path_filenames = sorted(get_file_name(path))
    labels, cluster_centers = kmeans_detect(path_filenames, 4)

    return cluster_centers

## save weight to .csv
def save_train(checkpoint):


    data = pd.DataFrame(data=checkpoint)
    data.to_csv('Model/dummy model.csv', encoding='gbk')

## model!!!
def dummy_model(img):
    cluster_centers = pd.read_csv('Model/dummy model.csv')
    cluster_centers = cluster_centers.values
    cluster_centers = np.delete(cluster_centers, 0, axis=1)

    labels_single, cluster_centers_single = kmeans_detect_single(img, 1)

    obj = cluster_centers_single[0]
    r = cluster_centers[0]
    d = cluster_centers[1]
    s = cluster_centers[2]
    o = cluster_centers[3]

    dist2r = np.sqrt(np.sum(np.square(obj - r)))
    dist2d = np.sqrt(np.sum(np.square(obj - d)))
    dist2s = np.sqrt(np.sum(np.square(obj - s)))
    dist2o = np.sqrt(np.sum(np.square(obj - o)))
    # print(dist2r,dist2d,dist2s,dist2o)

    sum = dist2o + dist2s + dist2d + dist2r
    dist2r_s = sum/dist2r
    dist2d_s = sum/dist2d
    dist2s_s = sum/dist2s
    dist2o_s = sum/dist2o

    sum_s = dist2o_s + dist2d_s + dist2s_s + dist2r_s


    rim = dist2r_s / sum_s
    dent = dist2d_s / sum_s
    scratch = dist2s_s / sum_s
    other = dist2o_s /sum_s


    return dent,other,  rim,scratch
# dent", "other", "rim", "scratch"
