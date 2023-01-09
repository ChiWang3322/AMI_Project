import json
import logging
import os
import shutil
import numpy as np
import torch

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = checkpoint
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist!")
    else:
        print("Checkpoint exists! ")
    torch.save(state, filepath)
    # if is_best:
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pt'))


def load_checkpoint(checkpoint):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("Checkpoint doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)

    return checkpoint
def mkdir(path):
    """
    Create path if not exist
    :param path: any path, e.g. "Data/"
    """
    if not os.path.exists(path):
        os.mkdir(path)

def split_dataset(path, train_percent=0.8, random_seed=20):
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