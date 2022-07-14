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
import pandas as pd
import numpy as np
import seaborn as sns
import time



def plot_prediction_results():
    rc = {'figure.figsize': (8, 4.5),
          'axes.facecolor': 'white',
          'axes.edgecolor': 'white',
          'axes.labelcolor': 'black',
          'figure.facecolor': 'white',
          'patch.edgecolor': 'white',
          'text.color': 'black',
          'xtick.color': 'black',
          'ytick.color': 'black',
          'grid.color': 'black',
          'font.size': 12,
          'axes.labelsize': 14,
          'xtick.labelsize': 12,
          'ytick.labelsize': 12}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    data_dict = {"Confidence": [0.7, 0.5, 0.3, 0.1, 0.1],
                 "Classes": ["rim", "dent", "scratch", "dot", "others"]}
    df_plot = pd.DataFrame().from_dict(data_dict)

    ax = sns.barplot(x="Classes", y="Confidence", data=df_plot, color="#b80606")
    y_str = "Confidence"
    x_str = "Classes"

    ax.set(xlabel=x_str, ylabel=y_str)
    ax.bar_label(ax.containers[0])

    st.pyplot(fig)

st.set_page_config(page_title="demo",layout="wide")

st.image("Development/icon.png", width=100)
# st.markdown("# 🚗")
st.title('Try Our Demo! 👋')
instructions = "Either upload your own image or select from the sidebar to get a preconfigured image."

description = "Here you can try:\n" \
              "- Our fine-tuned car damage classification models\n" \
              "- Label your own data\n" \
              "- Correct the classification results and store it in our database\n" \
              "Have a look at the sidebar for instruction. Let's get started, have fun! 😉" \
              "\n"

st.markdown(description)

# Choose model and actions
col1, col2 = st.columns(2)

st.sidebar.markdown('### Step 1.Choose model and action 🦸')
model_name = st.sidebar.selectbox(label = "Model", options = ["Dummy", "VGG16", "Resnet18", "Unet"])
action_type = st.sidebar.selectbox(
            "Action", ["None", "Prediction", "Labeling"])


st.sidebar.markdown("### Step 2.Choose or upload test imgae 🖼️")
genre = st.sidebar.radio(
     "I want to",
     ('Upload my own test image', 'Try pre-defined image database', ))

if genre == 'Try pre-defined image database':
     st.write('You selected comedy.')
else:
    file = st.sidebar.file_uploader("Upload your own image")

# if user uploaded file
if file:
    img = Image.open(file)
else:
    st.warning('You did not upload an image, a random image is selected.')
    img = Image.open("Development/Car.jpg")

resized_img = img.resize((300, 300))
col_img, col_pred = st.columns((.4, .6))
# Show image
with col_img:
    st.header("📷 Here is the image")
    st.image(resized_img)
    st.subheader(" The most probable type: rim")
with col_pred:
    st.header("📊 Prediction results in barplot ")
    plot_prediction_results()




st.markdown('\n')
st.header('Surprise! Enjoy our extra functions! 😁')
extra_funtion = st.selectbox("Extra functions", ["None", "Show our data distribution", "Show training process", "TBD"])

# if extra_funtion == "Show training process":
#     last_rows = np.array([0.1])
#     chart = st.line_chart(last_rows)
#
#     for i in range(1, 101):
#         new_rows =  np.array([i])
#         chart.add_rows(new_rows)
#         last_rows = new_rows
#         time.sleep(0.3)