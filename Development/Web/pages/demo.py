import json
# from io import BytesIO
from PIL import Image
import os
import matplotlib.pyplot as plt
# import boto3
# from botocore import UNSIGNED  # contact public s3 buckets anonymously
# from botocore.client import Config  # contact public s3 buckets anonymously
import pandas as pd
import streamlit as st
import numpy as np
import torch

precision_resnet ={'dent': 0.925925926, 'scratch': 0.85, 'other': 1.0, 'rim': 0.936170213}
recall_resnet = {'dent': 0.909090909, 'scratch': 0.871794872, 'other': 0.7, 'rim': 1}
total_resnet = 0.912162162

recall_vgg={'dent': 0.8181818181818182, 'scratch': 0.8205128205128205, 'other': 0.5, 'rim': 0.9772727272727273}
total_vgg= 0.8445945945945946
precision_vgg= {'dent': 0.8333333333333334, 'scratch': 0.7272727272727273, 'other': 1.0, 'rim': 0.9555555555555556}

recall_semi ={'dent': 0.711340206185567, 'scratch': 0.6986301369863014, 'other': 0.3333333333333333, 'rim': 0.975609756097561}
total_semi= 0.762962962962963
precision_semi= {'dent': 0.7582417582417582, 'scratch': 0.5730337078651685, 'other': 0.8571428571428571, 'rim': 0.963855421686747}

recall_active= {'dent':0.8363636363636363, 'scratch': 0.8974358974358975,'other': 0.7, 'rim': 0.9545454545454546 }
total_active= 0.8783783783783784
precision_active={'dent': 0.9019607843137255, 'scratch': 0.7291666666666666,'other': 1.0, 'rim':1.0}

def plot_two_bar(score1,score2,class_name):

    score1 = [float('{:.2f}'.format(i)) for i in score1]
    score2 = [float('{:.2f}'.format(i)) for i in score2]
    x = np.arange(len(class_name))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, score1, width, label='recall_rate',color='#fcc047')
    rects2 = ax.bar(x + width/2, score2, width, label='precision_rate',color='#6d4595')

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


st.set_page_config(page_title="demo",layout="wide")

# Overview
st.title("Overview of Our Demo 🚗")
# Model
st.markdown("### 1. Model")
with st.expander("✌ VGG "):
    # st.markdown("...")
    img = Image.open("Web/pages/images/vgg_architech.png")
    col1, col2, col3 = st.columns([2, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.image(img)

    with col3:
        st.write("")
    st.markdown("- VGG16 is a convolutional neural network model proposed by K. Simonyan and A. Zisserman "
                "from the University of Oxford in the paper “Very Deep Convolutional Networks "
                "for Large-Scale Image Recognition”.")
    st.markdown("- The model achieves 92.7% top-5 test accuracy in ImageNet, "
                "which is a dataset of over 14 million images belonging to 1000 classes.")
    st.markdown("- VGG16 is famous for it's simplicity and scalability, the layer size is only reduced by pooling layer.")
with st.expander("🚀 Resnet"):

    img = Image.open("Web/pages/images/ResNet-18-Architecture.png")
    col1, col2, col3 = st.columns([1, 6, 1])

    with col1:
        st.write("")

    with col2:
        st.image(img)

    with col3:
        st.write("")
    st.markdown("- Resnet models were proposed in “Deep Residual Learning for Image Recognition”.")
    st.markdown("- The model is proposed to solve untrainable problems for very deep ConvNets")
    st.markdown("- With skip connection, that is the main idea in Resnet, people can train very deep network even more than 180 layers")

# Techniques
st.markdown("### 2. Techniques")
row1, row3 = st.columns(2)
with row1:
    with st.expander("🕵️‍♀ Active Learning"):
        img = Image.open("Web/pages/images/ActiveLearning.png")
        st.image(img)
        st.markdown("- Active learning is a special case of machine learning in which a learning algorithm can interactively query a user (or some other information source) to label new data points with the desired outputs")
with row3:
    with st.expander("🏃‍♂ Semi-Supervised Learning"):
        img = Image.open("Web/pages/images/simclr_contrastive_learning.png")
        st.image(img)
        st.markdown("Semi-Supervised Learning(SSL) requires only a small amount of labeled data and do not need human labelilng during the training process. The SSL model we use is call SimCLR."
                    " It takes an unsupervised Contrastive Learning with unlabeled data to learn a representations of data and then use supervised learning to classify the representations with labeled data. ")





# Results
col1, col2 = st.columns(2)

st.header("Some Results of Our Work 👋")
# col_1, col_2 = st.columns((.4, .6))
# with col_1:
model_result = st.selectbox("Models", [ "VGG16", "Resnet18","Semi-supervised","Active learning"])
col1, col2 = st.columns(2)
vgg_result_path = "Web/pages/model_results/vgg16_result.pt"
resnet_result_path = "Web/pages/model_results/resnet18_result.pt"
if model_result == "VGG16":
    with col1:
        plot_two_bar(recall_vgg.values(), precision_vgg.values(), recall_vgg.keys())
        st.subheader("The total accuracy is {0:.2%}".format(total_vgg))
    with col2:
        vgg_result = torch.load(vgg_result_path,map_location ='cpu')
        fig, ax1 = plt.subplots()
        ax1.plot(vgg_result["train_acc"])
        ax1.plot(vgg_result["val_acc"])
        ax1.legend(['Training accuracy', 'Validation accuracy'])
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.grid(True)
        fig.tight_layout()
        st.pyplot(fig)
elif model_result == "Resnet18":
    # plot_two_bar()
    with col1:
        plot_two_bar(recall_resnet.values(), precision_resnet.values(), recall_resnet.keys())
        st.subheader("The total accuracy is {0:.2%}".format(total_resnet))
    with col2:
        resnet_result = torch.load(resnet_result_path,map_location ='cpu')
        fig, ax1 = plt.subplots()
        ax1.plot(resnet_result["train_acc"])
        ax1.plot(resnet_result["val_acc"])
        ax1.legend(['Training accuracy', 'Validation accuracy'])
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.grid(True)
        fig.tight_layout()
        st.pyplot(fig)
col4,col5,col6=st.columns((1,2,1))
if model_result == "Semi-supervised":
    # plot_two_bar()
    with col5:
        plot_two_bar(recall_semi.values(), precision_semi.values(), recall_semi.keys())
        st.subheader("  The total accuracy is {0:.2%}".format(total_semi))
elif model_result == "Active learning":
    # plot_two_bar()
    with col5:
        plot_two_bar(recall_active.values(), precision_active.values(), recall_active.keys())
        st.subheader("The total accuracy is {0:.2%}".format(total_active))
st.markdown('\n')
st.header('Enjoy Our Functions on The Left Side! 😁')

