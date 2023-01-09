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
def chang_model_dict_key(PATH):
    model = vgg16(pretrained=False, num_classes=4)
    print(torch.load(PATH, map_location=torch.device('cpu'))["model_state_dict"].keys())
    model_state_dict = torch.load(PATH, map_location=torch.device('cpu'))["model_state_dict"]
    key_list = model_state_dict.keys()
    new_model_state_dict = {}
    for key in key_list:
        print(key)
        new_model_state_dict['parent_module.'+key] = model_state_dict[key]
    torch.save(new_model_state_dict, './tmp/best_VGG.pt')

if __name__== "__main__":
    PATH = './tmp/best.pt'
    chang_model_dict_key(PATH)