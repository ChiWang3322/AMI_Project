import os
import shutil
import numpy as np


def mkdir(path):
    """
    Create path if not exist
    :param path: any path, e.g. "Data/"
    """
    if not os.path.exists(path):
        os.mkdir(path)


def split_dataset(path, train_percent, random_seed):
    """
    Spilt train_percent of images in path into train folder and (1-train_percent) of images into val folder
    Usage: split_dataset("Data/",0.8)
    :param path: The path of data, in which exists four folders: "dent","other","rim","scratch"
    :param train_percent: The proportion of training data,range: [0.0-1.0]
    :param random_seed: random seed for splitting dataset
    """
    np.random.seed(random_seed)
    name_list = ["dent", "other", "rim", "scratch"]
    mkdir(path+"train/")
    mkdir(path + "val/")
    for name in name_list:
        class_folder = path+name+"/"
        images = os.listdir(class_folder)

        mkdir(path + "train/" + name)  # create four folder of different classes
        train_images = np.random.choice(images, size=int(len(images)*train_percent), replace=False)
        for img in train_images:
            shutil.copyfile(class_folder+img, path+"train/"+name+"/"+img)

        mkdir(path+"val/"+name)  # create four folder of different classes
        for img in images:
            if img not in train_images:
                shutil.copyfile(class_folder+img, path+"val/"+name+"/"+img)
