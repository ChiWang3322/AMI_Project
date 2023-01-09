from preprocessing import transforms_1
from torchvision import datasets
import torch
import os
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.generators import nc_benchmark, ni_benchmark
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, \
    loss_metrics, timing_metrics, cpu_usage_metrics, confusion_matrix_metrics, disk_usage_metrics
from avalanche.models import SimpleMLP
from torchvision.models import vgg16
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
# ******************************
from avalanche.training.supervised import EWC
from torch.optim import Adam
import torch
import torch.nn as nn
from uncertainty import uncertainty

use_cuda = torch.cuda.is_available()


def data_load(transforms_1):
    '''
    Load data from path /Data/train and /Data/val and create dataloaders for each of them.
    Return the dataloaders, size of each dataloader and names of classes.
    Author: Jinghan Zhang
    '''
    data_transforms = transforms_1
    data_dir = "./Data/"
    image_datasets_train = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train'])
    image_datasets_val = datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    return image_datasets_train, image_datasets_val




if __name__=='__main__':
    image_datasets_train, image_datasets_val = data_load(transforms_1)

    scenario = ni_benchmark(
        image_datasets_train, image_datasets_val, n_experiences=10, shuffle=True, seed=1234,
        balance_experiences=True
    )
    print(scenario.n_classes)

    # MODEL CREATION
    model = vgg16(pretrained=True)
    # print(model)
    model.classifier._modules['6'] = nn.Linear(4096, scenario.n_classes)

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
        confusion_matrix_metrics(num_classes=scenario.n_classes, save_image=False,
                                 stream=True),
        disk_usage_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loggers=[interactive_logger, text_logger, tb_logger]
    )

    device = torch.device("cpu")

    optimizer = Adam(model.parameters(), lr=0.01)
    criterion = CrossEntropyLoss()

    cl_strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_mb_size=128,
        train_epochs=3,
        eval_mb_size=128,
        device=device,
        evaluator=eval_plugin,
        ewc_lambda=0.4,
    )

    # TRAINING LOOP
    print('Starting experiment...')
    results = []
    for experience in scenario.train_stream:
        # train returns a dictionary which contains all the metric values
        res = cl_strategy.train(experience)

        # test also returns a dictionary which contains all the metric values
        results.append(cl_strategy.eval(scenario.test_stream))
