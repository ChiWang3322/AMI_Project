import torch

model_checkpoint = "../Model/resnet18_new.pt"
checkpoint = torch.load(model_checkpoint)


del checkpoint["model"]

for key in ["train_acc", "val_acc"]:
    temp = []
    for value in checkpoint[key]:
        value = value.to('cpu')
        value = value.item()
        temp.append(value)
    checkpoint[key] = temp
for key in checkpoint.keys():
    print(checkpoint[key])
torch.save(checkpoint, "../Web/pages/model_results/resnet18_result.pt")

