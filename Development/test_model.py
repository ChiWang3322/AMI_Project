from model import *
from torchvision.models import vgg16, resnet18


def Acc_model(model_ft, dataloader):
    '''
    Return the accuracy and class-level accuracy
    example: (# predicted as rim and is true rim)/ (# all rim)
    author: Jinghan Zhang
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model_ft.eval()
    corrects = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}  # correct number for each class
    total = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}  # total number for each class
    recall_rate = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}  # recall rate for each class
    precision_rate = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}
    predict = {"dent": 0, "scratch": 0, "other": 0,
               "rim": 0}  # predict results from model(including predict true & false)
    acc_all = 0.0  # total accurate number
    total_all = 0  # total image number
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # outputs = model_ft.model(inputs)
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(preds)):
            total[class_names[labels.data[i]]] += 1
            total_all += 1
            predict[class_names[preds[i]]] += 1
            if preds[i] == labels.data[i]:  # if predict correctly
                acc_all += 1
                for target in ["dent", "scratch", "other", "rim"]:
                    if class_names[preds[i]] == target:
                        corrects[target] += 1
    for target in ["dent", "scratch", "other", "rim"]:
        recall_rate[target] = corrects[target] * 1.0 / total[target]
        precision_rate[target] = corrects[target] * 1.0 / predict[target] if predict[target]!=0 else 0

    total_acc = acc_all / total_all
    return recall_rate, corrects, total_acc, total_all, precision_rate

def Acc_semi_model(model_ft, dataloader):
    '''
    Return the accuracy and class-level accuracy
    example: (# predicted as rim and is true rim)/ (# all rim)
    author: Jinghan Zhang
    '''
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model_ft.eval()
    corrects = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}  # correct number for each class
    total = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}  # total number for each class
    recall_rate = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}  # recall rate for each class
    precision_rate = {"dent": 0, "scratch": 0, "other": 0, "rim": 0}
    predict = {"dent": 0, "scratch": 0, "other": 0,
               "rim": 0}  # predict results from model(including predict true & false)
    acc_all = 0.0  # total accurate number
    total_all = 0  # total image number
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model_ft.model(inputs)
        # outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        for i in range(len(preds)):
            total[class_names[labels.data[i]]] += 1
            total_all += 1
            predict[class_names[preds[i]]] += 1
            if preds[i] == labels.data[i]:  # if predict correctly
                acc_all += 1
                for target in ["dent", "scratch", "other", "rim"]:
                    if class_names[preds[i]] == target:
                        corrects[target] += 1
    for target in ["dent", "scratch", "other", "rim"]:
        recall_rate[target] = corrects[target] * 1.0 / total[target]
        precision_rate[target] = corrects[target] * 1.0 / predict[target]

    total_acc = acc_all / total_all
    return recall_rate, corrects, total_acc, total_all, precision_rate

class SimCLR(pl.LightningModule):

    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, 'The temperature must be a positive float!'
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(pretrained=False,
                                                   num_classes=4 * hidden_dim)  # Output of last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

def test_logreg(device,CHECKPOINT_PATH,batch_size, test_feats_data, **kwargs):
    # Data loaders
    test_loader = data.DataLoader(test_feats_data, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)
    trainer = pl.Trainer()
    pl.seed_everything(42)  # To be reproducable
    model = LogisticRegression.load_from_checkpoint(CHECKPOINT_PATH)
    # Test best model on train and validation set
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"]}
    return model, result
def predict_simCLR(img):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    simclr_model = SimCLR.load_from_checkpoint("./Model/Semi_Supervised.ckpt")
    network = deepcopy(simclr_model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)
    test_feats_simclr = network(img.to(device))
    model = LogisticRegression.load_from_checkpoint("simclr_Model/logr.ckpt")
    outputs = model.model(test_feats_simclr)
    return outputs

if __name__ == '__main__':

    batch_size = 1
    data_loaders, dataset_sizes, class_names = data_load(batch_size, transforms_1)
    model_name = "resnet18.pt"
    if model_name == "vgg.pt":
        model = torch.load("./Model/" + model_name, map_location=torch.device('cpu'))
        recall_rate, corrects, total_acc, total_all, precision_rate = Acc_model(model, data_loaders['val'])
        print("class recall_rate:", recall_rate, "total acc:", total_acc)
        print("class precision_rate:", precision_rate)
    elif model_name == "Semi_Supervised.ckpt":
        simclr_model = SimCLR.load_from_checkpoint("./Model/"+model_name)
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        test_img_data = datasets.ImageFolder("./Data/val", img_transforms)
        test_feats_simclr = prepare_data_features(device, simclr_model, test_img_data)
        # _, results = test_logreg(device, "./Model/logr.ckpt", batch_size=64,
        #                                          test_feats_data=test_feats_simclr,
        #                                          feature_dim=test_feats_simclr.tensors[0].shape[1],
        #                                          num_classes=4,
        #                                          lr=1e-3,
        #                                          weight_decay=1e-3)
        test_loader = data.DataLoader(test_feats_simclr, batch_size=batch_size, shuffle=False,
                                      drop_last=False, pin_memory=True, num_workers=0)
        trainer = pl.Trainer()
        model = LogisticRegression.load_from_checkpoint("simclr_Model/logr.ckpt")
        recall_rate, corrects, total_acc, total_all, precision_rate = Acc_semi_model(model, test_loader)
        print("class recall_rate:", recall_rate, "total acc:", total_acc)
        print("class precision_rate:",precision_rate)
    else:
        model = torch.load("./Model/" + model_name, map_location=torch.device('cpu'))
        recall_rate, corrects, total_acc, total_all,precision_rate = Acc_model(model, data_loaders['val'])
        print("class recall_rate:", recall_rate, "total acc:", total_acc)
        print("class precision_rate:",precision_rate)
