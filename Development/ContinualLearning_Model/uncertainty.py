from torchvision.models import vgg16
import torch
from PIL import Image
import torchvision.transforms as transforms
from baal.bayesian.dropout import MCDropoutModule
from torch import nn, optim
from baal.modelwrapper import ModelWrapper
from baal.utils.metrics import Accuracy
import os

USE_CUDA = False

def uncertainty(img_path):
    PATH = './tmp/best_VGG.pt'
    image = Image.open(img_path)
  
    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize(224),
                                transforms.RandomCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    img_tensor = transform(image)
    img_tensor = img_tensor.view(-1, 3, 224, 224)
    # print(img_tensor.size())
    model = vgg16(pretrained=False, num_classes=4)
    # print(model)
    # This will modify all Dropout layers to be usable at test time which is
    # required to perform Active Learning.
    model = MCDropoutModule(model)
    if USE_CUDA:
        model.cuda()
    criterion = nn.CrossEntropyLoss()

    # ModelWrapper is an object similar to keras.Model.
    baal_model = ModelWrapper(model, criterion, replicate_in_memory=False)
    baal_model.add_metric(name='accuracy',initializer=lambda : Accuracy())

    baal_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    output = baal_model.predict_on_batch(img_tensor.float(), cuda=False)
    output = torch.flatten(output)
    return output
    

if __name__=="__main__":
    img_path = './dent_00.jpeg'
    output = uncertainty(img_path)
    classes = os.listdir('./tmp/AMI')
    dict_classes = {str(i): classes[i] for i in range(len(classes))}
    print('the order of classes:', dict_classes)
    print(output)
    print(output.argmax())