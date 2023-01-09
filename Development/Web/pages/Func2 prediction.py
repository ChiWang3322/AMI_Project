import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil
import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from preprocessing import *
from utils import *
import dummy
from test_model import predict_simCLR
def model_list():
    model_library_l = os.listdir('Model')
    for i in range(len(model_library_l)):
        if model_library_l[i].endswith('.pt'):
            model_library_l[i] = model_library_l[i][:-3]
        elif model_library_l[i].endswith('.csv'):
            model_library_l[i] = model_library_l[i][:-4]
        elif model_library_l[i].endswith('.ckpt'):
            model_library_l[i] = model_library_l[i][:-5]
    return model_library_l


class_name = ["dent", "other", "rim", "scratch"]
uncertain_folder = "Data/uncertain/"
mkdir(uncertain_folder)
for model_name in model_list():
    mkdir(uncertain_folder+model_name)


def plot_prediction_results(score_list, class_name):
    score_list = [float('{:.3f}'.format(i)) for i in score_list]
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
          'font.size': 14,
          'axes.labelsize': 16,
          'xtick.labelsize': 14,
          'ytick.labelsize': 14}
    plt.rcParams.update(rc)
    fig, ax = plt.subplots()
    ### Goals
    data_dict = {"Confidence": score_list,
                 "Classes": class_name}
    df_plot = pd.DataFrame().from_dict(data_dict)

    ax = sns.barplot(x="Classes", y="Confidence", data=df_plot, color="#fcc047")
    y_str = "Confidence"
    x_str = "Classes"
    ax.spines['bottom'].set_color('#909090')
    ax.spines['left'].set_color('#909090')
    ax.set(xlabel=x_str, ylabel=y_str)
    ax.bar_label(ax.containers[0])

    st.pyplot(fig)

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Image Classification App")
st.write("")

st.markdown('### Step 1.Choose model 🦸')
model_name = st.selectbox(label = "Model", options = model_list(),index=model_list().index("Example Model"))
st.markdown("### Step 2.Choose or upload test imgae 🖼️")
genre = st.radio(
    "I want to",
    ('Try pre-defined image database', 'Upload my own test image'))

if genre == 'Try pre-defined image database':
    st.subheader('You selected pre-defined image.')
    file = "Web/pages/images/rim_pre-defined.jpeg"
elif genre == 'Upload my own test image':
    file = st.file_uploader("Upload your own image")


# if user uploaded file
if file:
    img = Image.open(file)
    st.write("")
    # st.write("Just a second...")
    img = Image.open(file)
    # inputs = torch.unsqueeze(transform(img), 0)
    print(model_name=='dummy model')
    if model_name == 'dummy model':
        scores = [(list(dummy.dummy_model(img)))]
        preds = np.argmax(scores)
    elif model_name == 'Semi_Supervised':
        outputs = predict_simCLR(torch.unsqueeze(transforms_1['val'](img), 0))
        m = torch.nn.Softmax(dim=1)
        outputs = m(outputs)
        scores = outputs.tolist()
        preds = np.argmax(scores)
    else:
        inputs = torch.unsqueeze(transforms_1['val'](img), 0)
        model = torch.load("Model/"+model_name+".pt",map_location ='cpu')
        if type(model) is dict:
            model = model["model"]
        model.eval()
        outputs = model(inputs)
        m = torch.nn.Softmax(dim=1)
        outputs = m(outputs)
        scores = outputs.tolist()

        _, preds = torch.max(outputs, 1)

    resized_img = img.resize((300, 300))
    col_img, col_pred = st.columns((.4, .6))
    # Show image
    with col_img:
        st.subheader("📷 Here is the image")
        st.image(resized_img)
        # st.write("")
        # st.subheader("The most probable type: {0}".format(class_name[preds]))
        st.markdown("<h5 style='text-align: left; '>The most probable type: {0}</h5>".format(class_name[preds]),
                    unsafe_allow_html=True)
    with col_pred:
        st.subheader("📊 Prediction results in barplot ")
        plot_prediction_results(scores[0], class_name)
    st.subheader(":disappointed_relieved: Not satisfied? - Let's mark it and retrain the model!")
    if st.button('Mark 👈'):
        if (genre == 'Try pre-defined image database'):
            shutil.copy(file, uncertain_folder+model_name)
        else:
            with open(os.path.join(uncertain_folder+model_name,file.name),"wb") as f:
                f.write(file.getbuffer())
        # os.remove(file)
        st.success('Marked successful! This image has been put into uncertain folder, You can re-label it and retrain our model in Func3')
else:
    st.warning('You did not upload an image, a random image is selected.')
    img = Image.open("Web/Car.jpg")
