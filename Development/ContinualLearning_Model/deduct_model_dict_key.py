from torchvision.models import vgg16
import torch
from PIL import Image
import torchvision.transforms as transforms
from baal.bayesian.dropout import MCDropoutModule
from torch import nn, optim
from baal.modelwrapper import ModelWrapper
from baal.utils.metrics import Accuracy
import os

'''
If keys do not match, using this function to change the name of keys of the state_dict 
stored inside of model.pt

Author: Zheng Tao
'''
def deduct_model_dict_key(model_dict):
    model = vgg16(pretrained=False, num_classes=4)
    model_state_dict = model_dict
    key_list = model_state_dict.keys()
    new_model_state_dict = {}
    for key in key_list:
        # print(key[14:])
        new_model_state_dict[key[14:]] = model_state_dict[key]
    # torch.save(new_model_state_dict, './tmp/best_VGG.pt')

    return new_model_state_dict

if __name__== "__main__":
    PATH = './tmp/best.pt'
    deduct_model_dict_key(PATH)