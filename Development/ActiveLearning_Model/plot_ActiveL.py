from glob import glob
import os
import random
import pandas as pd
# os.chdir('C:\\Users\\TaoZ\\Desktop\\baal')
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(__file__))
from baal.active import FileDataset, ActiveLearningDataset
from torchvision import transforms

import torch
from torch import nn, optim
from baal.modelwrapper import ModelWrapper
from torchvision.models import vgg16
from baal.bayesian.dropout import MCDropoutModule
# USE_CUDA = torch.cuda.is_available()
USE_CUDA = False
# print(USE_CUDA)
from baal.active.heuristics import BALD
from PIL import Image
import numpy as np
import subprocess
import matplotlib.pyplot as plt

from baal.utils.metrics import Accuracy

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# main_path = 'C:\\Users\\TaoZ\\Desktop\\baal'
import streamlit as st

# This function would do the work that a human would do.
def get_label(img_path, classes, type, mode):
    if mode == 'Y':
        ## without using web side
        if type == 'val':
            if '\\' in img_path:
                return classes.index(img_path.split('\\')[-2])
            else:
                return classes.index(img_path.split('/')[-2])
        elif type == 'train':
            img = Image.open(img_path)  # 打开图片，返回PIL image对象
            
            plt.figure(figsize=(4, 4))
            plt.ion()  # 打开交互模式
            plt.axis('off')  # 不需要坐标轴
            plt.imshow(img)
            plt.show()
        
            while True:
                receive = input("input:")
                index = int(receive)
                if index < len(classes):
                    plt.ioff()  # 显示完后一定要配合使用plt.ioff()关闭交互模式，否则可能出奇怪的问题
                    plt.clf()  # 清空图片
                    plt.close()  # 清空窗口
                    return index
                else:
                    continue

        ## with using web side
        # if type == 'val':
        #     if '\\' in img_path:
        #         return classes.index(img_path.split('\\')[-2])
        #     else:
        #         return classes.index(img_path.split('/')[-2])
        # elif type == 'train':
        #     img = Image.open(img_path)  # 打开图片，返回PIL image对象
        #     ## transfer to web
        #     st.image(img)
        #     # plt.figure(figsize=(4, 4))
        #     # plt.ion()  # 打开交互模式
        #     # plt.axis('off')  # 不需要坐标轴
        #     # plt.imshow(img)
        #     # plt.show()
        
        #     while True:
        #         ## receive from web
        #         img_type_l = ['','dent', 'other', 'rim', 'scratch']
        #         while True:
        #             receive_label = st.selectbox('label image',img_type_l)
        #             if receive_label != '':
        #                 receive = np.where(img_type_l == receive_label)
        #                 break
        #         # receive = input("input:")
        #         index = int(receive)
        #         if index < len(classes):
        #             plt.ioff()  # 显示完后一定要配合使用plt.ioff()关闭交互模式，否则可能出奇怪的问题
        #             plt.clf()  # 清空图片
        #             plt.close()  # 清空窗口
        #             return index
        #         else:
        #             continue
    else:
        if '\\' in img_path:
                return classes.index(img_path.split('\\')[-2])
        else:
            return classes.index(img_path.split('/')[-2])

def save_tO_csv(step, test_accuracy, train_accuracy):
    #将数据保存在一维列表
    list = [step, test_accuracy, train_accuracy]
    #由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv('./train_acc.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了






def Active_Learning(mode, train, val):
    # PATH = MOdel_PATH
    # files = glob('../../Data/*/*.jpeg')
    classes = ['dent', 'other', 'rim', 'scratch']
    dict_classes = {str(i): classes[i] for i in range(len(classes))}
    print('the order of classes:', dict_classes)
    # train, val = train_test_split(files, random_state=1337)  # Split 75% train, 25% validation
    # print(f"Train: {len(train)}, Valid: {len(val)}, Num. classes : {len(classes)}")


    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.Resize(224),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # We use -1 to specify that the data is unlabeled.
    train_dataset = FileDataset(train, [-1] * len(train), train_transform)

    val_transform = transforms.Compose([transforms.Resize(224),
                                        transforms.RandomCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    # We use -1 to specify that the data is unlabeled.
    val_dataset = FileDataset(val, [-1] * len(val), val_transform)
    active_learning_ds = ActiveLearningDataset(train_dataset, pool_specifics={'transform': val_transform})


    # model = vgg16(pretrained=True)
    # model.classifier._modules['6'] = nn.Linear(4096, 4)
    model = torch.load("C:/Users/TaoZ/Desktop/Group09/Development/Model/vgg16_new.pt", map_location='cpu')
    # This will modify all Dropout layers to be usable at val time which is
    # required to perform Active Learning.
    model = MCDropoutModule(model)
    if USE_CUDA:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    # ModelWrapper is an object similar to keras.Model.
    baal_model = ModelWrapper(model, criterion)
    baal_model.add_metric(name='accuracy',initializer=lambda : Accuracy())
    # baal_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    heuristic = BALD(shuffle_prop=0.1)


    # 1. Label all the val set and some samples from the training set.
    for idx in range(len(val_dataset)):
        img_path = val_dataset.files[idx]
        val_dataset.label(idx, get_label(img_path, classes, 'val', mode))

    # Let's label 100 training examples randomly first.
    # Note: the indices here are relative to the pool of unlabelled items!
    train_idxs = np.random.permutation(np.arange(len(train_dataset)))[:10].tolist()
    labels = []
    for idx in train_idxs:
        print(train_dataset.files[idx])
        labels.append(get_label(train_dataset.files[idx], classes, 'train', mode))
    active_learning_ds.label(train_idxs, labels)

    print(f"Num. labeled: {len(active_learning_ds)}/{len(train_dataset)}")

    test_accuracy = []
    train_accuracy = []
    for step in range(100): # 5 Active Learning step!
        # 2. Train the model for a few epoch on the training set.
        print(f"Training on {len(active_learning_ds)} items!")
        baal_model.train_on_dataset(active_learning_ds, optimizer, batch_size=16, epoch=1, use_cuda=USE_CUDA)
        baal_model.test_on_dataset(val_dataset, batch_size=16, use_cuda=USE_CUDA)

        output_metrics = {k:v.avg for k,v in baal_model.metrics.items()}
        print("Metrics:", output_metrics)
        # print("Metrics:", baal_model.get_metrics())

        test_accuracy.append(output_metrics["test_accuracy"])
        train_accuracy.append(output_metrics["train_accuracy"])
        save_tO_csv(step, test_accuracy, train_accuracy)
        # 3. Select the K-top uncertain samples according to the heuristic.
        pool = active_learning_ds.pool
        if len(pool) == 0:
            print("We're done!")
            break
        elif output_metrics['train_accuracy'] > 0.9:
            print("Network is well trained")
            break
        predictions = baal_model.predict_on_dataset(pool, batch_size=16, iterations=1, use_cuda=USE_CUDA, verbose=False)
        top_uncertainty = heuristic(predictions)[:10]
        # 4. Label those samples.
        oracle_indices = active_learning_ds._pool_to_oracle_index(top_uncertainty)
        labels = []
        for idx in oracle_indices:
            labels.append(get_label(train_dataset.files[idx], classes, 'train', mode))
        active_learning_ds.label(top_uncertainty, labels)

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(test_accuracy,label="val")
    plt.plot(train_accuracy,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    return baal_model
    # torch.save(baal_model.state_dict(), '../Model/VGG16_cl.pt')


if __name__ == '__main__':
    # Model_PATH = 'C:/Users/TaoZ/Desktop/Group09/Development/Model/Base_VGG16.pt'
    # mode = input('want to annotate:(Y/N) \t') # Y: want to annotate
    df = pd.DataFrame(columns=['step','test_accuracy','train_accuracy'])#列名
    df.to_csv("./train_acc.csv",index=False) #路径可以根据需要更改

    files = glob('C:/Users/TaoZ/Desktop/Group09/Development/Data/val/*/*.jpeg')
    train, val = train_test_split(files, random_state=1337)  # Split 75% train, 25% validation
    print(f"Train: {len(train)}, Valid: {len(val)}, Num. classes : {4}")
    Active_Learning("N", train, val)