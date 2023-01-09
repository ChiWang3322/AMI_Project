from PIL import Image
import streamlit as st
import os
import shutil
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import dataLoad
from model import *
from glob import glob
from Development.ContinualLearning_Model.CL_selfdefined_dataset import continual_learning
def mkdir(path):
    """
    Create path if not exist
    :param path: any path, e.g. "Data/"
    """
    if not os.path.exists(path):
        os.mkdir(path)

uncertain_folder = "Data/uncertain/"  # this is the folder to save uncertain images
retrain_folder = "Data/retrain/"
retrain_labeled_folder = "Data/retrain_labeled/"
mkdir(uncertain_folder)
mkdir(retrain_folder)
mkdir(retrain_labeled_folder)

def retrain_count(labeled_folder):
    retrain_num: int = 0
    for folder in os.listdir(labeled_folder):
        retrain_num += len(os.listdir(labeled_folder + '/' + folder))
    return retrain_num

img_type_l = ['dent', 'other', 'rim', 'scratch']


def model_list():
    model_library_l = os.listdir('Model')
    model_l=[]
    for i in range(len(model_library_l)):
        if model_library_l[i].endswith('.pt'):
            model_l.append(model_library_l[i][:-3])
        # elif model_library_l[i].endswith('.csv'):
        #     del model_library_l[i]

    return model_l



def empty_retrain(retrain_folder):
    for img in os.listdir(retrain_folder):
        os.remove(retrain_folder + img)

def retrain_label_count(labeled_folder):
    retrain_num=[0]*4
    for idx,folder in enumerate(os.listdir(labeled_folder)):
        if folder == 'train' or folder == 'val':
            continue
        retrain_num[idx] = len(os.listdir(labeled_folder + '/' + folder))
    return retrain_num
def Acc_model(model_ft, dataloader, class_names):
    '''
    Return the accuracy and class-level accuracy
    example: (# predicted as rim and is true rim)/ (# all rim)
    author: Jinghan Zhang
    ADD:08.23,2022;one more argument 'class_names' is added cuz class_names is not a global var here
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft.eval()
    corrects = {"dent": 0, "rim": 0, "scratch": 0, "other": 0}  # correct number for each class
    total = {"dent": 0, "rim": 0, "scratch": 0, "other": 0}  # total number for each class
    recall_rate = {"dent": 0, "rim": 0, "scratch": 0, "other": 0}  # recall rate for each class
    precision_rate = {"dent": 0, "rim": 0, "scratch": 0, "other": 0}
    precict = {"dent": 0, "rim": 0, "scratch": 0, "other": 0
               }  # predict results from model(including predict true & false)
    acc_all = 0.0  # total accurate number
    total_all = 0  # total image number
    Percent = 0
    totalDataPercent = 1.0/(len(dataloader))
    my_bar = st.progress(0.0)
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)

        Percent = totalDataPercent + Percent
        if(Percent>0.99):
            Percent = 0.99
        # print(Percent)
        my_bar.progress(Percent)

        for i in range(len(preds)):
            total[class_names[labels.data[i]]] += 1
            total_all += 1
            precict[class_names[preds[i]]] += 1
            if preds[i] == labels.data[i]:  # if predict correctly
                acc_all += 1
                for target in ["dent", "scratch", "other", "rim"]:
                    if class_names[preds[i]] == target:
                        corrects[target] += 1
    #     Percent = totalDataPercent + Percent
    #     if(Percent>0.99):
    #         Percent = 0.99
    # # print(Percent)
    #     my_bar.progress(Percent)
    for target in ["dent", "scratch", "other", "rim"]:
        recall_rate[target] = corrects[target] * 1.0 / total[target]
        precision_rate[target] = corrects[target] * 1.0 / precict[target]

    total_acc = acc_all / total_all
    # print(total_all)
    my_bar.progress(1.0)
    return recall_rate, corrects, total_acc, total_all, precision_rate


def plot_two_bar(score1,score2,class_name):

    score1 = [float('{:.2f}'.format(i)) for i in score1]
    score2 = [float('{:.2f}'.format(i)) for i in score2]
    x = np.arange(len(class_name))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, score1, width, label='new model',color='#fcc047')
    rects2 = ax.bar(x + width/2, score2, width, label='old model',color='#6d4595')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Percent')
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x, class_name)
    ax.legend()
    ax.spines['bottom'].set_color('#909090')
    ax.spines['left'].set_color('#909090')
    ax.bar_label(rects1, padding=3,label_type='center')
    ax.bar_label(rects2, padding=3,label_type='center',color='white')

    fig.tight_layout()

    # plt.show()
    st.pyplot(fig)


st.set_page_config(page_title="Correction", layout="wide")

# st.image("Web/icon.png", width=100)
st.title('Correction ✔️')

description = "Not satisfied with our model? Now you can:\n" \
              "- Choose the uncertain images you want to retrain.\n" \
              "- Label them by yourself.\n" \
              "- Retrain the model with Continual Learning\n" \
               \
              "\n"
model_retrain = st.selectbox('Choose your model to update:', model_list(),index=model_list().index("Example Model"))

st.markdown(description)

func = st.radio(
    "FUNCTION",
    ('Add uncertain images to retrain', 'Labeling', 'Retrain'))

uncertain_folder_model = uncertain_folder + model_retrain + '/'
mkdir(uncertain_folder_model)
retrain_folder_model = retrain_folder + model_retrain + '/'
retrain_labeled_folder_model = retrain_labeled_folder + model_retrain + '/' ##
mkdir(retrain_folder_model)
mkdir(retrain_labeled_folder_model)
for img_type in img_type_l:
    # mkdir(retrain_folder_model+img_type)
    mkdir(retrain_labeled_folder_model + img_type)
if func == 'Add uncertain images to retrain':
    st.markdown('## Add uncertain images to retrain')
    uncertain_list = os.listdir(uncertain_folder_model)
    if len(uncertain_list) >= 1:
        img_num = st.slider('choose the number of uncertain images:', 0, len(uncertain_list), 1)
        if img_num > 0:
            col = [0, 1, 2]
            container = [0] * img_num
            col[0], col[1], col[2] = st.columns(3)
            # uncertain_list = os.listdir(uncertain_folder)
            for i in range(img_num):
                col_idx = i % 3
                img = Image.open(uncertain_folder_model + uncertain_list[i])
                img = img.resize((224, 224))
                container[i] = col[col_idx].container()
                container[i].image(img)

            if st.button("Add to retrain library"):
                for i in range(img_num):
                    shutil.copyfile(uncertain_folder_model + uncertain_list[i], retrain_folder_model + uncertain_list[i])
                    os.remove(uncertain_folder_model + uncertain_list[i])

        retrain_list = os.listdir(retrain_folder_model)
        st.text("All retrain images:")
        col_2 = st.columns(20)
        if len(retrain_list) > 0:
            for i in range(len(retrain_list)):
                img = Image.open(retrain_folder_model + retrain_list[i])
                img = img.resize((224, 224))
                col_2[i % 20].image(img)
        if st.button('clear'):
            empty_retrain(retrain_folder_model)
    else:
        st.warning("There is no uncertain images for " + model_retrain + " now. Please assign uncertain images in previous page Func2.")

if func == 'Labeling':
    st.markdown('## Labeling')

    retrain_list = os.listdir(retrain_folder_model)
    if len(retrain_list) == 0:
        st.warning("There is no images for " + model_retrain + " to label now. Please add some uncertain images")
    img_st = st.empty()
    if len(retrain_list) > 0:
        st.info('Number of images to be labeled:' + str(len(os.listdir(retrain_folder_model))), icon="ℹ️")

        img = Image.open(retrain_folder_model + retrain_list[0])  # images to add label
        img_st.image(img, width=400)
        label = st.selectbox('label image', img_type_l)

    # button.button('confirm')
        if st.button('confirm'):
            shutil.copyfile(retrain_folder_model + retrain_list[0], retrain_labeled_folder_model + label+ '/' + retrain_list[0])
            os.remove(retrain_folder_model + retrain_list[0])
            retrain_list = os.listdir(retrain_folder_model)
            if len(retrain_list) > 0:
                img = Image.open(retrain_folder_model + retrain_list[0])
                img_st.image(img, width=400)

if func == 'Retrain':
    st.markdown('## Retrain')
    st.text("Added " + str(retrain_count(retrain_labeled_folder_model)) + " images to retrain")
    # if retrain_count(retrain_labeled_folder_model) == 0:
    #     st.text("You should assign at least one image to retrain!")
    if 0 in retrain_label_count(retrain_labeled_folder_model) or 1 in retrain_label_count(retrain_labeled_folder_model):
        st.warning("You should assign at least 2 image for each class!")
        st.text("Number of images for each class in retrain folder")
        for classname in ['dent', 'other', 'rim', 'scratch']:
            st.text('- {}:{}'.format(classname, len(os.listdir(retrain_labeled_folder_model+classname))))
    else:
        col_3 = st.columns(4)
        for i,img_type in enumerate(img_type_l):
            col_3[i].text(img_type)
            retrain_labeled_list = os.listdir(retrain_labeled_folder_model+img_type)
            if len(retrain_labeled_list) > 0:
                for img in retrain_labeled_list:
                    im = Image.open(retrain_labeled_folder_model+img_type+'/'+img)
                    im = im.resize((96, 96))
                    col_3[i].image(im)
        if st.button('retrain'):
            st1=st.empty()
            st1.text('Retraining ' + model_retrain + ', you can take a break now!')
            # retrain model
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            label_list = ['dent', 'other', 'rim', 'scratch']
            dirpath = retrain_labeled_folder_model
            print('place images into %s' % dir)
            files = glob(dirpath+'/*/*.jpeg')
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
            save_path = "Model/"
            torch.save(new_model, save_path + '/' + model_retrain + "_retrained.pt")
            # clear all trained data
            shutil.rmtree(retrain_labeled_folder_model)
            os.mkdir(retrain_labeled_folder_model)

            old_model = torch.load("Model/"+model_retrain+".pt",map_location ='cpu')

            st1.text('Retrain finished, enjoy your new model!')
            # remove images from /retrain_labeled

            st.subheader(":100: - Let's see the model criterion!")
            batch_size = 1
            st.write("Test Progress:")
            data_loaders, dataset_sizes, class_names = dataLoad.data_load(batch_size, transforms_1)
            # my_bar.progress(25)
            st.write("Testing New Model:")
            recall_rate, corrects, total_acc_new, total_all, precision_rate_new = Acc_model(new_model, data_loaders['val'],
                                                                                    class_names)
            st.write("Comparing with old model Model:")
            _, _, total_acc_old, _, precision_rate_old = Acc_model(old_model, data_loaders['val'],
                                                                                    class_names)
            # recall_rate['total acc'] = total_acc

            plot_two_bar(precision_rate_new.values(), precision_rate_old.values(), precision_rate_new.keys())
            st.subheader("The accuracy is {0:.2%}".format(total_acc_new))
            st.subheader("The accuracy increased {0:.2%}".format(total_acc_new-total_acc_old))

