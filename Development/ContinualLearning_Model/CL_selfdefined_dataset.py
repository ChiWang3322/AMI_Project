from preprocessing import transforms_1
from torchvision import datasets, transforms
import torch
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
#from uncertainty import uncertainty
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from ContinualLearning_Model.deduct_model_dict_key import deduct_model_dict_key
from utils import split_dataset

use_cuda = torch.cuda.is_available()

def data_load_CL(transforms_1, Model_name):
    data_transforms = transforms_1
    data_dir = "./Data/retrain_labeled/"+Model_name+'/'
    # data_dir = "C:/Users/TaoZ/Desktop/Group09/Development/Data/"
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
    PATH = './Model/'+ Model_name+'.pt'
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

    image_datasets_train, image_datasets_val = data_load_CL(transforms_1, Model_name)

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

    # cl_strategy = Naive(
    #                     model, 
    #                     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #                     CrossEntropyLoss(), 
    #                     train_mb_size=500, 
    #                     train_epochs=1, 
    #                     eval_mb_size=100,
    #                     evaluator=eval_plugin
    #                     )
    
    # cl_strategy = SynapticIntelligence(
    #                     model, 
    #                     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #                     CrossEntropyLoss(), 
    #                     si_lambda = 0.5,
    #                     eps = 1e-7,
    #                     train_mb_size=500, 
    #                     train_epochs=1, 
    #                     eval_mb_size=100,
    #                     device = 'cpu',
    #                     evaluator=eval_plugin,
    #                     eval_every = 1,
    #                     )

    cl_strategy = Cumulative(
                        model, 
                        SGD(model.parameters(), lr=0.001, momentum=0.9),
                        CrossEntropyLoss(), 
                        train_mb_size=500, 
                        train_epochs=1, 
                        eval_mb_size=100,
                        device = 'cpu',
                        evaluator=eval_plugin,
                        eval_every = 1,
                        )

    # cl_strategy = Replay(
    #                     model, 
    #                     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #                     CrossEntropyLoss(), 
    #                     mem_size= 200,
    #                     train_mb_size=500, 
    #                     train_epochs=1, 
    #                     eval_mb_size=100,
    #                     device = 'cpu',
    #                     evaluator=eval_plugin,
    #                     eval_every = 1,
    #                     )

    # cl_strategy = GDumb(
    #                     model, 
    #                     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #                     CrossEntropyLoss(), 
    #                     mem_size= 200,
    #                     train_mb_size=500, 
    #                     train_epochs=1, 
    #                     eval_mb_size=100,
    #                     device = 'cpu',
    #                     evaluator=eval_plugin,
    #                     eval_every = 1,
    #                     )

    # cl_strategy = LwF(
    #                     model, 
    #                     SGD(model.parameters(), lr=0.001, momentum=0.9),
    #                     CrossEntropyLoss(), 
    #                     alpha = 1,
    #                     temperature = 2,
    #                     train_mb_size=500, 
    #                     train_epochs=1, 
    #                     eval_mb_size=100,
    #                     device = 'cpu',
    #                     evaluator=eval_plugin,
    #                     eval_every = 1,
    #                     )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)

        # train returns a dictionary which contains all the metric values
        # res = cl_strategy.train(experience)
        print('Training completed')

        print('Computing accuracy on the whole test set')
        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream))

    return model



if __name__=='__main__':
    # dirpath = "C:\\Users\\TaoZ\\Desktop\\Avalance\\CL_data"
    # print('place images into %s' % dir)
    # N_image = int(input('how many images stored: \t'))
    # input_task_labels = [0]*N_image
    # for i in range(N_image):
    #     input_task_labels[i] = int(input('task_label: \t'))
    # continual_learning(dirpath, input_task_labels)
    model_retrain = 'vgg16'
    # C:\\Users\\TaoZ\\Desktop\\Group09\\
    dirpath = './Data/retrain_labeled' + model_retrain + '/'
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
    # C:/Users/TaoZ/Desktop/Group09/
    torch.save(new_model, './Model/'+ model_retrain+'_retrain.pt')