from preprocessing import *
from torchvision import datasets
import torch
import os


def data_load(batch_size, transforms):
    '''
    Load data from path /Data/train and /Data/val and create dataloaders for each of them.
    Return the dataloaders, size of each dataloader and names of classes.
    Author: Jinghan Zhang
    '''
    data_transforms = transforms
    data_dir = "Data/"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return data_loaders, dataset_sizes, class_names

def data_load_retrain(batch_size, transforms, data_dir):
    '''
    Load data from path /Data/retrain create the dataloader.
    Return the dataloader, size of dataloader and names of classes.
    Author: Jinghan Zhang
    '''
    data_transforms = transforms
    # data_dir = "Data/retrain_labeled"
    image_datasets = {x: datasets.ImageFolder(data_dir, data_transforms[x]) for x in ['train']}
    data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=1) for x in ['train']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
    class_names = image_datasets['train'].classes
    return data_loaders, dataset_sizes, class_names
