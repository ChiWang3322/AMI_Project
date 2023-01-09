from torchvision import datasets
import os

def get_dataset(data_dir, data_transforms):

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    return image_datasets