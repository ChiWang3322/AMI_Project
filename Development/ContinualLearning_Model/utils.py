import os
import shutil
import numpy as np
import json
from collections import defaultdict
import datetime

def mkdir(path):
    """
    Create path if not exist
    :param path: any path, e.g. "Data/"
    """
    if not os.path.exists(path):
        os.mkdir(path)


def split_dataset(path, train_percent, random_seed):
    """
    Spilt train_percent of images in path into train folder and (1-train_percent) of images into val folder
    Usage: split_dataset("Data/",0.8)
    :param path: The path of data, in which exists four folders: "dent","other","rim","scratch"
    :param train_percent: The proportion of training data,range: [0.0-1.0]
    :param random_seed: random seed for splitting dataset
    """
    np.random.seed(random_seed)
    name_list = ["dent", "other", "rim", "scratch"]
    mkdir(path+"train/")
    mkdir(path + "val/")
    for name in name_list:
        class_folder = path+name+"/"
        images = os.listdir(class_folder)

        mkdir(path + "train/" + name)  # create four folder of different classes
        train_images = np.random.choice(images, size=int(len(images)*train_percent), replace=False)
        for img in train_images:
            shutil.copyfile(class_folder+img, path+"train/"+name+"/"+img)

        mkdir(path+"val/"+name)  # create four folder of different classes
        for img in images:
            if img not in train_images:
                shutil.copyfile(class_folder+img, path+"val/"+name+"/"+img)


def data_label(path,mode):
    '''
    Create json file for labeled or to-be labeled data
    :param path: the path where 4 label file exist, e.g: './data'.  
    :param mode: 1. create json file from original existed data
                 2. create a new jsonfile with timestamp
    import:
        import os
        import json
        from collections import defaultdict
        import datetime
        from os.path import join as pjoin
    '''
    label_4=['rim','dent','scratch','other']
    
    if mode==1:
        data_list=defaultdict(list)
        data_list={'annotations':[]}

        for label in label_4: # rim scratch dent other
            pic_path=path+'/'+label
            for filename in os.listdir(pic_path):   
                data_list['annotations'].append({'filename':filename,'label':label})

        json_file='./'+path+'data_labeled.json' # output jsonfile
        with open(json_file,'w') as f:
            f.write(json.dumps(data_list,ensure_ascii=False,indent=2))

    elif mode==2:
        data_list=defaultdict(list)
        data_list={'annotations':[]}
        
        nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') #
        timestamp_str=str(nowTime)
        
        for label in label_4: # rim scratch dent other
            pic_path=path+'/'+label
            for filename in os.listdir(pic_path):   
                data_list['annotations'].append({'filename':filename,'label':label})

        json_file='./'+path+'data_label'+'_'+timestamp_str+'.json' # output jsonfile
        print(json_file)
        with open(json_file,'w') as f:
            f.write(json.dumps(data_list,ensure_ascii=False,indent=2))
