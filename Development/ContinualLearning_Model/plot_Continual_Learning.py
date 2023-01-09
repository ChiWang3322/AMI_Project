from preprocessing import transforms_1
from torchvision import datasets, transforms
import torch
import pandas as pd
import os
from glob import glob
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from avalanche.benchmarks.generators import filelist_benchmark, dataset_benchmark, \
                                            tensors_benchmark, paths_benchmark
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from torchvision.models import vgg16, resnet18
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, CWRStar, LFL, Cumulative, StreamingLDA, Replay, GSS_greedy, GDumb, LwF,SynapticIntelligence
# ******************************
from avalanche.training.supervised import EWC
from torch.optim import Adam
import torch
import torch.nn as nn
from uncertainty import uncertainty
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from deduct_model_dict_key import deduct_model_dict_key
from utils import split_dataset
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()

def save_tO_csv(step, test_accuracy, train_accuracy):
    #将数据保存在一维列表
    list = [step, test_accuracy, train_accuracy]
    #由于DataFrame是Pandas库中的一种数据结构，它类似excel，是一种二维表，所以需要将list以二维列表的形式转化为DataFrame
    data = pd.DataFrame([list])
    data.to_csv('./train_acc.csv',mode='a',header=False,index=False)#mode设为a,就可以向csv文件追加数据了

def data_load(transforms_1, Model_name):
    data_transforms = transforms_1
    data_dir = "C:/Users/TaoZ/Desktop/Group09/Development/Data/data_supplement/"
    # data_dir = "C:/Users/TaoZ/Desktop/Group09/Development/Data/retrain_labeled/"+Model_name+'/'
    random_seed = 112233
    train_percent = 0.8
    split_dataset(data_dir, train_percent, random_seed)
    image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    image_datasets_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    return image_datasets_train, image_datasets_val

def continual_learning(dirpath, input_task_labels, Model_name):
    '''
    images are stored in rel_dir and wait to sent into continual learning

    if structlog version conflict, place install structlog==21.1.0
    author: Zheng Tao
    '''
    # PATH = 'C:/Users/TaoZ/Desktop/Group09/Development/Model/'+ Model_name+'.pt'
    PATH = 'C:/Users/TaoZ/Desktop/Group09/Development/Model/'+ Model_name+'.pt'
    # train_experiences = []
    # for rel_dir, label in zip(
    #         ["dent", "other", "rim", "scratch"],
    #         [0, 1, 2, 3]):
    #     # First, obtain the list of files
    #     filenames_list = os.listdir(os.path.join(dirpath, rel_dir))

    #     if len(filenames_list) > 0:
    #         experience_paths = []
    #         for name in filenames_list:
    #             instance_tuple = (os.path.join(dirpath, rel_dir, name), label)
    #             experience_paths.append(instance_tuple)
    #         train_experiences.append(experience_paths)
    #     else:
    #         continue
    # # Here we create a GenericCLScenario ready to be iterated
    # scenario = paths_benchmark(
    #     train_experiences,
    #     [train_experiences[0][0], train_experiences[1][0], train_experiences[2][0], train_experiences[3][0]],  # Single test set
    #     task_labels=input_task_labels,
    #     complete_test_set_only=True,
    #     train_transform=transforms.ToTensor(),
    #     eval_transform=transforms.ToTensor()
    # )

    image_datasets_train, image_datasets_val = data_load(transforms_1, Model_name)

    scenario = ni_benchmark(
        image_datasets_train, image_datasets_val, n_experiences=10, shuffle=True, seed=1234,
        balance_experiences=True
    )

    # MODEL CREATION
    if Model_name.startswith('vgg'):
        # model = vgg16(pretrained=False, num_classes=4)
        # # model.classifier._modules['6'] = nn.Linear(4096, 4)
        # # model_dict = torch.load(PATH, map_location=torch.device('cpu'))
        # model_dict = torch.load(PATH, map_location=torch.device('cpu'))["model_state_dict"]
        # for key in model_dict.keys():
        #     if key.startswith('parent_module'):
        #         model_dict = deduct_model_dict_key(model_dict)
        #         break
        #     else:
        #         break
        # model.load_state_dict(model_dict)
        model = torch.load(PATH, map_location=torch.device('cpu'))
    else:
        model = torch.load(PATH, map_location=torch.device('cpu'))

    for name, param in model.named_parameters():
        print(name)
        # print(param.data)
        # print("requires_grad:", param.requires_grad)
        # print("-----------------------------------")

    # log to Tensorboard
    tb_logger = TensorboardLogger()

    # log to text file
    text_logger = TextLogger(open('log.txt', 'a'))

    # print to stdout
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, epoch_running=True),
        forgetting_metrics(experience=True, stream=True),
        cpu_usage_metrics(experience=True),
        confusion_matrix_metrics(num_classes=4, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    device = torch.device("cpu")

    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()

    # cl_strategy = EWC(
    #                     model=model,
    #                     optimizer=optimizer,
    #                     criterion=criterion,
    #                     train_mb_size=500,
    #                     train_epochs=1,
    #                     eval_mb_size=100,
    #                     device=device,
    #                     evaluator=eval_plugin,
    #                     ewc_lambda=0.2,
    #                     mode = "separate",
    #                 )
    cl_strategy = SynapticIntelligence(
                        model, 
                        SGD(model.parameters(), lr=0.001, momentum=0.9),
                        CrossEntropyLoss(), 
                        si_lambda = 0.5,
                        eps = 1e-7,
                        # alpha = 1,
                        # temperature = 2,
                        # mem_size= 200,
                        # output_layer_name = 'classifier.6',
                        # lambda_e = 0.0016,
                        # cwr_layer_name = 'classifier.6',
                        # input_size = [3, 224, 224],
                        # num_classes = 4,
                        train_mb_size=500, 
                        train_epochs=1, 
                        eval_mb_size=100,
                        device = 'cpu',
                        evaluator=eval_plugin,
                        eval_every = 1,
                        # shrinkage_param = 0.5，
                        # streaming_update_sigma = True，
                    )

    # TRAINING LOOP
    print('Starting experiment...')
    train_resultS = []
    test_resultS = []
    for step, experience in enumerate(scenario.train_stream):
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        train_result = cl_strategy.train(experience)
        print('Training completed！！！！！！！！！！！！！！！！！！！！！！！')
        print(train_result)

        # test also returns a dictionary which contains all the metric values
        # results.append(cl_strategy.eval(scenario.test_stream))
        test_result = cl_strategy.eval(scenario.test_stream)
        print('Computing accuracy on the whole test set###################')
        print(test_result)

        test_resultS.append(test_result["Top1_Acc_Stream/eval_phase/test_stream/Task000"])
        train_resultS.append(train_result["Top1_Acc_Epoch/train_phase/train_stream/Task000"])
        save_tO_csv(step, test_resultS, train_resultS)

    plt.figure(figsize=(10,5))
    plt.title("Training and Validation accuracy")
    plt.plot(test_resultS,label="val")
    plt.plot(train_resultS,label="train")
    plt.xlabel("iterations")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    return model



if __name__=='__main__':
    # dirpath = "C:\\Users\\TaoZ\\Desktop\\Avalance\\CL_data"
    # print('place images into %s' % dir)
    # N_image = int(input('how many images stored: \t'))
    # input_task_labels = [0]*N_image
    # for i in range(N_image):
    #     input_task_labels[i] = int(input('task_label: \t'))
    # continual_learning(dirpath, input_task_labels)
    model_retrain = 'vgg16_new'
    dirpath = 'C:/Users/TaoZ/Desktop/Group09/Development/Data/data_supplement/'
    print('place images into %s' % dirpath)
    print(dirpath+'/*/*.jpeg')
    files = glob(dirpath+'/*/*.jpeg')
    label_list = ['dent', 'other', 'rim', 'scratch']
    N_image = len(files)
    input_task_labels = []
    for i in range(N_image):
        single_image_path = files[i]
        if '\\' in single_image_path:
            label = single_image_path.split('\\')[-2]
        else:
            label = single_image_path.split('/')[-2]
        input_task_labels.append(label_list.index(label))
    new_model = continual_learning(dirpath, input_task_labels, model_retrain)
    torch.save(new_model, 'C:/Users/TaoZ/Desktop/Group09/Development/Model/'+ model_retrain+'_retrain.pt')