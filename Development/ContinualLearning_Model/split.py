from utils import split_dataset

random_seed = 112233
path = './Data/'
train_percent = 0.8
split_dataset(path, train_percent, random_seed)
