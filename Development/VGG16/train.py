from utils import *
from build_dataset import *
from build_dataloader import *
import torch

import torch.optim as optim
from torch.optim import lr_scheduler
from model.mymodel import *
import numpy as np
import time
import os
import copy
from utils import split_dataset

# Check if original data is split
ori_data_path = "../Data/original/"
if not os.path.exists(ori_data_path + "train"):
    split_dataset(ori_data_path, train_percent=0.8)


params = {"num_epochs": 40,
          "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
          "model_checkpoint": "./vgg16checkpoint",
          "num_classes": 4,
          "pretrained": True,
          "lr":0.0001,
          "patience": 5,
          "batch_size":16
          }

datasets = get_dataset(data_dir=ori_data_path, data_transforms=get_transforms())
dataloaders, dataset_sizes, _ = get_dataloaders(datasets, batch_size=params["batch_size"])


def train_model(model, criterion, optimizer, scheduler, params):
    since = time.time()
    num_epochs = params["num_epochs"]
    model_checkpoint = params["model_checkpoint"]
    patience_count = 0
    last_val_loss = 0
    if not os.path.exists(model_checkpoint):
        print("[INFO]: Model checkpoin doesn't exist {}".format(model_checkpoint))
        best_acc = 0.0
        print("[INFO]: Current best accuracy:", best_acc)
        train_loss = []
        val_loss = []
        train_acc = []
        val_acc = []
    else:
        try:
            checkpoint = torch.load(model_checkpoint + "/best.pt")
            print("[INFO]: Loading checkpoint successfully, initializing...")
            best_acc = checkpoint["best_acc"]
            print("[INFO]: Current best accuracy:", best_acc)
            train_loss = checkpoint["train_loss"]
            val_loss = checkpoint["val_loss"]
            train_acc = checkpoint["train_acc"]
            val_acc = checkpoint["val_acc"]
        except:
            print("Permission denied, initializing...")
            best_acc = 0.0
            print("Current best accuracy:", best_acc)
            train_loss = []
            val_loss = []
            train_acc = []
            val_acc = []

    best_model_wts = copy.deepcopy(model.state_dict())
    device = params["device"]

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 30)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                # inputs = inputs.to(device)
                # labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.to(device))
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.to(device))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.to(device).data)
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            if phase == 'train':
                scheduler.step()
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)
            elif phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

                # Early stopping strategy
                if epoch_loss > last_val_loss:
                    patience_count += 1
                    if patience_count >= params["patience"]:
                        print("Reaching patience, early stopping...")
                        model.load_state_dict(best_model_wts)
                        return model
                else:
                    patience_count = 0
                print("Current patience count:", patience_count)
                last_val_loss = epoch_loss
            # deep copy the model
            if phase == 'val':
                best_acc = epoch_acc

                # save checkpoint
                state = {'epoch': epoch,
                            'model': model,
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'train_acc': train_acc,
                            'val_acc': val_acc,
                            'best_acc':best_acc,
                            }
                save_checkpoint(state, True, model_checkpoint)
                print("Epoch finished, checkpoint saved")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # Creat model
    model_checkpoint = "../Model/vgg16.pt"
    # model_checkpoint = "../Model/resnet18_new.pt"

    # model_checkpoint = "./efficientnetb5_checkpoint"


    params["model_checkpoint"] = model_checkpoint
    if not os.path.exists(model_checkpoint):
        model = get_vgg16(params)
        # model = get_efficientnetb5(params)
        print("[INFO]: Creating pretrained model successfully, start training using:", params["device"])
    else:
        checkpoint = torch.load(model_checkpoint)
        model = checkpoint["model"]
        print("[INFO]: Loading best checkpoint successfully, start training using:", params["device"])
    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Define optimizer
    optimizer_ft = optim.Adam(model.parameters(), lr=params["lr"])

    # Define scheduler
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)


    # To device(GPU)
    model.to(params["device"])

    # Training
    model = train_model(model, criterion, optimizer_ft, exp_lr_scheduler, params)
