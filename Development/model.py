from torchvision import models
import torchvision
from torchvision import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from train import *
from dataLoad import *
from preprocessing import *
from utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from simclr import*
from ActiveLearning_Model.Active_Learning import Active_Learning
from glob import glob
from sklearn.model_selection import train_test_split
from dummy import *
import os, codecs
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image

def model_train(model: str):
    if model == "resnet18":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = 4
        data_loaders, dataset_sizes, class_names = data_load(batch_size, transforms_1)
        save_path = "Model/"
        mkdir(save_path)
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 4)
        model_ft = model_ft.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, data_loaders, device, dataset_sizes, num_epochs=25)
        torch.save(model_ft, save_path+'resnet18.pt')
    elif model == "simclr":
        # to be fulfilled
        CHECKPOINT_PATH = "simclr_Model/"
        NUM_WORKERS = 0
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        mkdir(CHECKPOINT_PATH)
        traindata_dir = "Data/train"
        valdata_dir = "Data/val"
        unlabeled_data = datasets.ImageFolder(os.path.join(traindata_dir),
                                              ContrastiveTransformations(contrast_transforms, n_views=2))
        train_data_contrast = datasets.ImageFolder(os.path.join(valdata_dir),
                                                   ContrastiveTransformations(contrast_transforms, n_views=2))
        simclr_model = train_simclr(CHECKPOINT_PATH,device,unlabeled_data,train_data_contrast,batch_size=256,
                                    hidden_dim=128,
                                    lr=5e-4,
                                    temperature=0.07,
                                    weight_decay=1e-4,
                                    max_epochs=500)
        train_img_data = datasets.ImageFolder(os.path.join(traindata_dir), img_transforms)
        test_img_data = datasets.ImageFolder(os.path.join(valdata_dir), img_transforms)
        train_feats_simclr = prepare_data_features(device,simclr_model, train_img_data)
        test_feats_simclr = prepare_data_features(device,simclr_model, test_img_data)
        simclr_log_model, results = train_logreg(device,CHECKPOINT_PATH,batch_size=64,
                                  train_feats_data=train_feats_simclr,
                                  test_feats_data=test_feats_simclr,
                                  feature_dim=train_feats_simclr.tensors[0].shape[1],
                                  num_classes=4,
                                  lr=1e-3,
                                  weight_decay=1e-3)
        print(results)
        # torch.save(simclr_log_model, CHECKPOINT_PATH + 'simclr_log.ckpt')

    elif model == "activel":
        files = glob('Data/*/*.jpeg')
        train, test = train_test_split(files, random_state=1337)  # Split 75% train, 25% validation
        print(f"Train: {len(train)}, Valid: {len(test)}, Num. classes : {4}")
        # 'Y' we want to annotate the dataset
        new_model = Active_Learning("N", train, test)
        torch.save(new_model, './Model/vgg16.pt')

     # elif model == "continuall":
    #     Model_path = './Model/VGG16_cl.pt'
    #     dirpath = "Somewhere to store unlabeled images"
    #     print('place images into %s' % dir)
    #     ## get from web N_image
    #     N_image = int(input('how many images stored: \t'))
    #     input_task_labels = [0]*N_image
    #     for i in range(N_image):
    #         ## get label from web input_task_labels
    #         input_task_labels[i] = int(input('task_label: \t'))
    #     model_dict = continual_learning(Model_path, dirpath, input_task_labels)
    #     torch.save(model_dict, './Model/VGG16_cl.pt')

    elif model == "dummy":
        ## get path of the training data
        path = 'Data/train'
        ## get the weight of the model
        dummy_weight = train_dummy(path, 4)
        ## save the weight
        save_train(dummy_weight)

    return





