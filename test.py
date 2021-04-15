import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
from numpy import array
import os
import PIL
from PIL import Image, ImageOps
import seaborn as sns
import pickle
from PIL import *
import gdown
import base64

with open('detection.json', 'r') as json_file:
    json_savedModel= json_file.read()

with open('emotion.json', 'r') as json_file:
    json_savedModel2= json_file.read()
    
    
try:
    model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)
    model_1_facialKeyPoints.load_weights('weights_keypoint.hdf5')
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model_1_facialKeyPoints.compile(loss="mean_squared_error", optimizer= adam , metrics = ['accuracy'])
    model_2_emotion = tf.keras.models.model_from_json(json_savedModel2)
    model_2_emotion.load_weights('weights_emotions.hdf5')
    model_2_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])
except:
    url = 'https://drive.google.com/u/2/uc?id=1O0q60wdUVcFKoY4n8EFSlUBWuQwMwYDI&export=download'
    output = 'weights_keypoint.hdf5'
    gdown.download(url, output, quiet=False) 
    model_1_facialKeyPoints = tf.keras.models.model_from_json(json_savedModel)
    model_1_facialKeyPoints.load_weights('weights_keypoint.hdf5')
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
    model_1_facialKeyPoints.compile(loss="mean_squared_error", optimizer= adam , metrics = ['accuracy'])
    model_2_emotion = tf.keras.models.model_from_json(json_savedModel2)
    model_2_emotion.load_weights('weights_emotions.hdf5')
    model_2_emotion.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])



# main_bg = "images.jpg"
# main_bg_ext = "jpg"

# side_bg = "images.jpg"
# side_bg_ext = "jpg"
def main():

    
    st.markdown("<h1 style='text-align: center; color: white;'>Emotion AI</h1>", unsafe_allow_html=True)
    st.write("")
#     st.markdown(
#         f"""
#         <style>
#         .reportview-container {{
#             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
#         }}
#     .sidebar .sidebar-content {{
#             background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
#         }}
#         </style>
#         """,
#         unsafe_allow_html=True
#     )
    

    file_up = st.file_uploader("Upload an image", type="jpg")
    if st.button("Predict"):
       img = Image.open(file_up)
       im2 = ImageOps.grayscale(img)
       im3 = im2.resize((96,96), Image.ANTIALIAS)
       im3.save('somepic.jpg')
       gray_img = array(im3)
       dummy = gray_img
       dummy = np.stack(dummy, axis = 0)
       dummy = dummy.reshape(1, 96, 96, 1)
       dummy = dummy/255
       df_predict_test = predict(dummy)
       if df_predict_test['emotion'].values.item() == 0:
         st.text("Angry")
       elif df_predict_test['emotion'].values.item() == 1:
         st.text("Disgust")
       elif df_predict_test['emotion'].values.item() == 2:
         st.text("Sad")
       elif df_predict_test['emotion'].values.item() == 3:
         st.text("Happy")
       elif df_predict_test['emotion'].values.item() ==4:
         st.text("Surprise")
         



    







def predict(X_test):

  # Making prediction from the keypoint model
  df_predict = model_1_facialKeyPoints.predict(X_test)

  # Making prediction from the emotion model
  df_emotion = np.argmax(model_2_emotion.predict(X_test), axis=-1)

  # Reshaping array from (856,) to (856,1)
  df_emotion = np.expand_dims(df_emotion, axis = 1)

  # Converting the predictions into a dataframe
  df_predict = pd.DataFrame(df_predict)

  # Adding emotion into the predicted dataframe
  df_predict['emotion'] = df_emotion

  return df_predict





   


if __name__ == '__main__':
	main()
