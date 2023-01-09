import requests
import streamlit as st
from streamlit_lottie import st_lottie
from PIL import Image
from streamlit_player import st_player

# Find more emojis here: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Our Webpage", page_icon=":tada:", layout="wide")


def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Use local CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
def show_video(url):


    c1, c3,c2 = st.columns([1,6,1])

    with c2:
        st.subheader(" ")
        # st.subheader("Parameters")
        options = {
            # "progress_interval": 1000,
            # "volume": st.slider("Volume", 0.0, 1.0, 1.0, .01),
            # "playing": st.checkbox("Playing", False),
            # "loop": st.checkbox("Loop", False),
            # "muted": st.checkbox("Muted", False),
        }

    #         with st.expander("SUPPORTED PLAYERS", expanded=True):
    #             st.write("""
    #             - Dailymotion
    #             - Facebook
    #             - Mixcloud
    #             - SoundCloud
    #             - Streamable
    #             - Twitch
    #             - Vimeo
    #             - Wistia
    #             - YouTube
    #             <br/><br/>
    #             """, unsafe_allow_html=True)

    with c3:
        event = st_player(url, **options, key="youtube_player")
    #         st.write(event)
    with c1:
        st.write(' ')


local_css("Web/pages/style/style.css")

# ---- LOAD ASSETS ----
lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
img_cover = Image.open("Web/pages/images/video_cover.png")
img_intro = Image.open("Web/pages/images/Data-Analytics.png")
# img_team_logo = Image.open("Web/pages/images/team-logo2.jpg")
video_file = open('Web/pages/project_video/001.mp4', 'rb')
lottie_contact = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_u25cckyh.json")

# ---- HEADER SECTION ----
with st.container():
    # left,right,last = st.columns((3,3,3))
    # with left:
    #     st.subheader("Hi, We are Eye Man(group9) :wave:")
    #
    # with right:
    #     # st.write(" ")
    #     st.image(img_team_logo,width=135)
    # with last:
    #     st.write(" ")
    left_text_column, right_image_column = st.columns((2,1))
    with left_text_column:
        st.subheader("Hi, We are Eye Man(group9) :wave:")
        st.title("Data Analysts From TUM")
        st.write("""
             - We want to **build a classification model** that can distinguish four different class damages.
             - We need to **process the already labeled images** to make them suitable for the classification task.
             - And to **enrich the dataset** to meet the huge demand for labeled images.
             - We also use **active learning** and **semi-supervised learning** to **tackle insufficient-labeled images situation**.
             """, unsafe_allow_html=True)

        # st.write("[Learn More >](https://pythonandvba.com)")
    with right_image_column:
        st.image(img_intro)
    # with middle_image:


# ---- WHAT We DO ----
with st.container():
    st.write("---")
    left_column, right_column = st.columns([6,3])
    with left_column:
        st.header("What we do")
        #         st.write("##")
        #         st.write(
        #             """
        #             we are creating intelligent models to classify car damage for WENN:
        #             - Use **ResNet18** and **VGG16** model to classify car damages.
        #             - Use **active learning** and **semi-supervised learning** methods to tackle labelless problems.
        #             - **Docker** -- make the code run, not just "on my PC", but also on yours.
        #             - A brilliant video and sensible webpages show our effort and results.

        #                 """
        #         )
        st.write("""We are creating intelligent models to classify car damage for WENN:""")
        tab1, tab2, tab3 = st.tabs(["MODELS 🧩", "Idea ✨", "Powered by 🎉"])

        with tab1:
            st.write("""
             - VGG16
             - ResNet18
             Use **VGG16** and **ResNet18** models to classify car damages.
             """, unsafe_allow_html=True)

        with tab2:
            st.write("""
            - Active Learning
            - Semi-Supervised Learning
            Use **active learning** and **semi-supervised learning** methods to tackle insufficient-labeled problem.
            """, unsafe_allow_html=True)

        with tab3:
            st.write("""
            - Streamlit
            - Docker
             Make the code run, not just "on my PC", but also on yours.
            """, unsafe_allow_html=True)

    #         with st.expander("OUR MODELS 🧩 ", expanded=False):
    #             st.write("""
    #             - VGG16
    #             - ResNet18
    #             Use **ResNet18** and **VGG16** model to classify car damages.
    #             """, unsafe_allow_html=True)
    #         with st.expander("Idea ✨ ", expanded=False):
    #             st.write("""
    #             - Active Learning
    #             - Semi-Supervised Learning
    #             Use **active learning** and **semi-supervised learning** methods to tackle insufficient-labele problem.
    #             """, unsafe_allow_html=True)
    #         with st.expander("Powered by 🎉 ", expanded=False):
    #             st.write("""
    #             - Streamlit
    #             - Docker
    #             - Gitlab
    #              Make the code run, not just "on my PC", but also on yours.
    #             """, unsafe_allow_html=True)

    with right_column:
        st_lottie(lottie_coding, height=300, key="coding")

# ---- PROJECTS ----
with st.container():
    st.write("---")
    st.header("Our Project Video")
    #     st.write("##")
    #     image_column, text_column = st.columns((1, 2))

    #     st.subheader("How our model distinguish car damage")
    #     st.write(
    #         """
    #         In this project, we......
    #
    #         """
    #     )
    #     st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")
    show_video("https://youtu.be/p8eMKibjeWw")

    # video_bytes = video_file.read()
    #
    # st.video(video_bytes)
    #     with image_column:
    #         st_player("https://youtu.be/TXSOitGoINE")
    #     with text_column:
    #         st.subheader("How our model distinguish car damage")
    st.write(
        """
        In this project, we use ResNet18 and VGG16 model to classify four different car damage!
        We also use active learning and semi-supervised learning to improve model accuracy!
        
        """
    )
#         st.markdown("[Watch Video...](https://youtu.be/TXSOitGoINE)")

# ---- CONTACT ----
with st.container():
    st.write("---")
    st.header("Get in Touch with Us!")
    st.write("##")

    # Documention: https://formsubmit.co/ !!! CHANGE EMAIL ADDRESS !!!
    contact_form = """
    <form action="https://formsubmit.co/tang.ziyou@tum.de" method="POST">
        <input type="hidden" name="_captcha" value="false">
        <input type="text" name="name" placeholder="Your name" required>
        <input type="email" name="email" placeholder="Your email" required>
        <textarea name="message" placeholder="Your message here" required></textarea>
        <button type="submit">Send</button>
    </form>
    """
    left_column, right_column = st.columns((2,1))
    with left_column:
        st.markdown(contact_form, unsafe_allow_html=True)
    with right_column:
        st_lottie(lottie_contact, height=200, key="contact")
# ---- Our Teams ----
with st.container():
    st.write("---")
    st.header("Our Teams")
    st.write("##")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image("Web/pages/images/Teammember photo/Yan Gao.jpg",width=132)
        # st.subheader("A cat")
        st.markdown("<h4 style='text-align: left; '>Yan Gao</h4>", unsafe_allow_html=True)
    with col2:
        st.image("Web/pages/images/Teammember photo/Chenghao Hu2.jpg",width=132)
        st.markdown("<h4 style='text-align: left; '>Chenghao Hu</h4>", unsafe_allow_html=True)
    with col3:
        st.image("Web/pages/images/Teammember photo/Xue Zhang.jpg",width=132)
        st.markdown("<h4 style='text-align: left; '>Xue Zhang</h4>", unsafe_allow_html=True)
    with col4:
        st.image("Web/pages/images/Teammember photo/Yiming Shan.jpg",width=132)
        st.markdown("<h4 style='text-align: left; '>Yiming Shan</h4>", unsafe_allow_html=True)
    with col5:
        st.image("Web/pages/images/Teammember photo/Ziyou Tang.jpg",width=132)
        st.markdown("<h4 style='text-align: left; '>Ziyou Tang</h4>", unsafe_allow_html=True)

    col5, col6, col7, col8, col9, col10 = st.columns([1, 4, 4, 4, 4, 1])
    # with col5_5:
    #     st.write(' ')
    with col6:
        st.image("Web/pages/images/Teammember photo/Zheng Tao.jpg",width=132)
        # st.subheader("  A cat")
        # st.markdown('  **A cat**')
        st.markdown("<h4 style='text-align: left; '>Zheng Tao</h4>", unsafe_allow_html=True)
    with col7:
        st.image("Web/pages/images/Teammember photo/Chi Wang2.jpg",width=132)
        st.markdown("<h4 style='text-align: left; '>Chi Wang</h4>", unsafe_allow_html=True)
    with col8:
        st.image("Web/pages/images/Teammember photo/Jinghan Zhang.jpg",width=132)
        st.markdown("<h4 style='text-align: left; '>Jinghan Zhang</h4>", unsafe_allow_html=True)
    with col9:
        st.image("Web/pages/images/Teammember photo/Kaicheng Ni.jpg",width=132)
        st.markdown("<h4 style='text-align: left; '>Kaicheng Ni</h4>", unsafe_allow_html=True)
    # with col10:
    #     st.write(' ')
