import streamlit as st
from PIL import Image
import os
import sys
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils import *
def mkdir(path):
    """
    Create path if not exist
    :param path: any path, e.g. "Data/"
    """
    if not os.path.exists(path):
        os.mkdir(path)
label_folder = "Data/labeled/"
mkdir(label_folder)

img_type_l = ['', 'dent', 'other', 'rim', 'scratch']

for img_type in img_type_l:
    mkdir(label_folder+img_type)

st.title("Labeling tool")
st.write("")

st.markdown("### label new images 🖼️")
st.markdown("- You can upload multiple images at the same time")




st.write("")
file_l = st.file_uploader("Upload your own image",accept_multiple_files=True)

# if user uploaded file
if file_l:
    label_l = [None] * len(file_l)

    for idx,file in enumerate(file_l):
        img = Image.open(file)
        st.image(img)
        label_l[idx] = st.selectbox('label image', img_type_l, key=idx)
        # if st.button('confirm',key=idx) and label_l[idx] != '':
        #     img.save(label_folder + label_l[idx] + '/' + file_l[idx].name)
    if '' not in label_l:
        st.markdown("### Step 1.")
        if st.button('Save label information and Export as JSON'):
            for idx, file in enumerate(file_l):
                img = Image.open(file)
                img.save(label_folder + label_l[idx] + '/' + file_l[idx].name)
                data_label(label_folder,mode=2)
            st.success("Done! ✅")
            # Download file
            output_filename = "./Data/labeled_data"
            shutil.make_archive(output_filename, 'zip', label_folder)
            with open(output_filename + ".zip", "rb") as zp:
                st.markdown("### Step 2.")
                download_button = st.download_button(
                    label="Download labeled data as zip file",
                    data=zp,
                    file_name='labeled_data.zip',
                    mime='application/zip',
                )

            shutil.rmtree(label_folder)
            mkdir(label_folder)
            os.remove(output_filename + ".zip")
            # st.markdown("### Step 3.")
            # if st.button('Empty existing labeled files'):
            # file_l = False


else:
    st.warning('You have not currently uploaded any image.')



