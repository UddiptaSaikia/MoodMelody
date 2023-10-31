import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from streamlit_option_menu import option_menu
import cv2
from keras.models import model_from_json
import numpy as np
import av
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode
import webbrowser
import os

PAGE_CONFIG={"page_title":"Mood-Melody","page_icon":"icons8-music-64.png","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)


# load model 
emotion_dict = {0 : 'angry', 1: 'happy', 2: 'sad', 3:'neutral'}
# load json and create model
json_file = open('final_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# load weights into new model
classifier.load_weights("final_model.h5")



try:
    emotion = np.load("Emotion.npy")
except:
    emotion=""

# feature extraction to feed the model    

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1,48,48,1)
    return feature/255.0

#load face using cv2 haarcascade 
try:
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ___________face detection using cv2 and prediting and storing the emotion for further proceedings________#

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            image=img_gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(
                x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                roi = extract_features(roi_gray)
                prediction = classifier.predict(roi)[0]
                maxindex = int(np.argmax(prediction))
                finalout = emotion_dict[maxindex]
                output = str(finalout)    # emotion stored as a string
                np.save("Emotion.npy",np.array(output)) # emotion string saved in a local file
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img
    
# ________________UI part of the application____________________#

def streamlit_menu():

#__________side menu__________#

    with st.sidebar:
        selected = option_menu(
            menu_title="Main Menu",  # required
            options=["Home","About", "Contact"],  # required
            icons=["house", "book", "envelope"],  # optional
            menu_icon="cast",  # optional
            default_index=0,  # optional
                
            styles={
                "container":
                {

                "background-color":"rgb(210,117,114)",
                },

                "nav-link" :{
                    "font-family":"monospace",
                    # "color" : "black",
                    # "background-color":"#AFF1FF",
                    # "--hover-color":"#D0F1F9",
                },
                "nav-link-selected" :{
                    "font-family":"monospace",
                    # "color": "black",
                },
                "menu-title":
                {
                     "font-family":"monospace",
                }
            },
        
                
        )
    return selected
    
selected = streamlit_menu()  

#_______text fields and buttons for singer,language_________#

if selected=="Home":

    st.markdown("""<h1>Instructions:</h1>""",unsafe_allow_html=True)
    st.markdown("""1.Choose the language from the list <br>
                2.Write the desired singer name<br>
                3.Press the enter/Recommand me songs button and then press start button<br>
                4.Look properly to your webcam and press Recommand me song button<br>
                <b> Enjoy your favourite music</b> <br>
                    <b>Warning</b> : In case you are getting the same emotion everytime, press RESET button and then try <br><br>""",unsafe_allow_html=True)

    language=st.selectbox("Choose language",('','English','Hindi','Assanese','Bengali','Others'))
    if language=='Others':
        language=st.text_input('Enter your choice')
    singer=st.text_input("Write singer name(Use proper singer name)")

    if language and singer:
        webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
                            video_processor_factory=Faceemotion)

    btn =st.button("RECOMMEND SONGS")

    # if btn :
    #     webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION,
    #                          video_processor_factory=Faceemotion)

    # btn2 =st.button("Voice Search")
    btn3=st.button("RESET")
    if btn3:
        try:
            if emotion=="":
                st.warning("Enter neccessery details first")
            else:
                os.remove("Emotion.npy") 
        except:
            os.remove("Emotion.npy") 



    #__________accessing the webbrowser for recomandation of songs___________#
    if btn:
        if not (emotion):
            st.warning("Let me capture your image")
        else:
            
            webbrowser.open(f'https://www.youtube.com/results?search_query={language}+{emotion}+song+{singer}')
            
    # if btn3:
    #     os.remove("Emotion.npy")        

#________css file for customisation_________#
with open ("style.css") as f:
    st.markdown(f'<style>{f.read()} </style>',unsafe_allow_html=True)

if selected=="About":
    st.markdown("""
                              <div>
            <h1>About the application</h1>
            <p>This project aims to create an Emotion-Based Music Recommender System using Convolutional Neural Networks (CNNs) and Transfer Learning. The system takes input from a webcam to detect the user's emotion and recommends music based on the detected emotion, favorite artist, and language preferences.</p>
            <h1 id="tech_uses">Used technologies and other resources</h1><br><br>
            <p id="tech_desc">1.CNN of Deep learning for the emotion detection model <br>
                2.FER13 data set for the training and testing of the CNN model<br>
                3.Different python libraries for the development of the model<br>
                4.Streamlit web application for deployment <br>
                5.HTML and CSS for styling the UI part <br>
                6.Webrtc API<br>
                7.Youtube as the song platform
                </p>
                

               
          </div>""",unsafe_allow_html=True)
    
if selected=="Contact":
    st.markdown("""<h1>Developer Details:</h1><br>
                    <p><b>Chinmoy Bora</b><br>
                <a href="https://www.linkedin.com/in/chinmoy-bora"><img width="48" height="48" src="https://img.icons8.com/fluency/48/linkedin.png" alt="linkedin"/></a>
                <a href="https://github.com/Chinmoy-Bora"><img width="64" height="64" src="https://img.icons8.com/sf-black-filled/64/github.png" alt="github"/></a>
                 <a href="https://instagram.com/chinmoy_cb_?utm_source=qr&igshid=MzNlNGNkZWQ4Mg%3D%3D"><img width="48" height="48" src="https://img.icons8.com/fluency/48/instagram-new.png" alt="instagram-new"/></a><br><br>
                <b> Gouranga Borah</b> <br>
                <a href="https://www.linkedin.com/in/gouranga-borah-87b080249"><img width="48" height="48" src="https://img.icons8.com/fluency/48/linkedin.png" alt="linkedin"/></a>
                <a href="https://github.com/b-Gouranga"><img width="64" height="64" src="https://img.icons8.com/sf-black-filled/64/github.png" alt="github"/></a>
                <a href="https://instagram.com/gouranga_borah_?igshid=NGVhN2U2NjQ0Yg=="><img width="48" height="48" src="https://img.icons8.com/fluency/48/instagram-new.png" alt="instagram-new"/></a><br><br>
                 <b>Uddipta Saikia</b> <br>
                <a href="https://www.linkedin.com/in/uddipta-saikia-7b21a3258"><img width="48" height="48" src="https://img.icons8.com/fluency/48/linkedin.png" alt="linkedin"/></a>
                <a href="https://github.com/UddiptaSaikia"><img width="64" height="64" src="https://img.icons8.com/sf-black-filled/64/github.png" alt="github"/></a>
                <a href="https://instagram.com/i_am_uddipta_?igshid=NzZlODBkYWE4Ng=="><img width="48" height="48" src="https://img.icons8.com/fluency/48/instagram-new.png" alt="instagram-new"/></a><br><br>
                <b> Keshab Sen</b><br>
                 <a href="https://www.linkedin.com/in/keshab-sen-a82b90251"><img width="48" height="48" src="https://img.icons8.com/fluency/48/linkedin.png" alt="linkedin"/></a>
                <a href="https://github.com/Keshab002?tab=repositories"><img width="64" height="64" src="https://img.icons8.com/sf-black-filled/64/github.png" alt="github"/></a>
                <a href="https://instagram.com/its_spidey02_?igshid=YTQwZjQ0NmI0OA=="><img width="48" height="48" src="https://img.icons8.com/fluency/48/instagram-new.png" alt="instagram-new"/></a><br><br>
                </p>

                    """,unsafe_allow_html=True)
    
    







