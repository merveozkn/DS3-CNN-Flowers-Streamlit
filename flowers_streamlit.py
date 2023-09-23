import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import requests
from io import BytesIO
st.markdown(
    """
    <style>
    .stApp {
        background-color: #D9FO8absF ;  
    </style>
    """,
    unsafe_allow_html=True
)
# Streamlit uygulamasını başlat

st.markdown('<div style="display: flex; justify-content: flex-end; margin-top:-70px"><img src="https://media.giphy.com/media/X5PsaxTP6U3h9dUSxd/giphy.gif" alt="GIF" width="100%" style="max-width: 200px; margin-right: 250px;"></div>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #658626; color: white; font-size: 30px; padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.1);">🌻Flower Prediction App🌻</p>', unsafe_allow_html=True)
st.markdown('<p style="background-color: #8FB447; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌼 Types of Flowers 🌼</p>', unsafe_allow_html=True)
st.image("Türler.png", use_column_width=True)

# Kullanıcıdan resim yükleme yöntemini seçmesini isteyin
st.sidebar.title("Image Upload Method")
upload_method = st.sidebar.radio("Please select a model:", ["Install from your computer", "Install with Internet Connection"])

uploaded_image = None  # Kullanıcının yüklediği resmi saklamak için

if upload_method == "Install from your computer":
    # Kullanıcıdan resim yükleme
    #st.write("Lütfen bir çiçek resmi yükleyin:")
    uploaded_image = st.file_uploader("Please upload a flower image:", type=["jpg", "png", "jpeg"])
elif upload_method == "Install with Internet Connection":
    # Kullanıcıdan internet linki alın
    st.write("Please enter the official internet link of a flower:")
    image_url = st.text_input("Image Link")

# Model seçimi
st.sidebar.title("Model Selection")
selected_model = st.sidebar.radio("Please select a model:", ["CNN_model", "VGG16_model", "ResNet_model", "Xception_model", "NASNetMobile_model","mobilenet_model"])


# Resmi yükle ve tahmin et butonları
if uploaded_image is not None or (upload_method == "Install with Internet Connection" and image_url):
    st.markdown('<p style="background-color: #8FB447; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌼Image of your choice🌼</p>', unsafe_allow_html=True)
    #st.write("Seçtiğiniz Resim")
    if uploaded_image is not None:
        st.image(uploaded_image, caption='', use_column_width=True)
    elif upload_method == "Install with Internet Connection" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='', use_column_width=True)
        except Exception as e:
            st.error("There was an error uploading the image. Please enter a valid internet link.")

# Model bilgisi düğmesi
if st.sidebar.button("Information about the Model"):
    st.markdown(f'<p style="background-color: #8a4baf; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌷{selected_model}🌷</p>', unsafe_allow_html=True)

    if selected_model == "CNN_model":
        st.write("CNN_model, is a basic Convolutional Neural Network (CNN) model. It includes convolutional layers, pooling layers and fully connected layers. It is often used for basic visual classification tasks.")
    elif selected_model == "VGG16_model":
        st.write("VGG16_model,It is a 16-layer deep Convolutional Neural Network model. It contains alternating convolutional and pooling layers. It is used for tasks such as visual classification and object recognition.") 
    elif selected_model == "ResNet_model":
        st.write("ResNet_model is a deep Convolutional Neural Network model that uses 'residual' blocks to make it easier to train deep networks. It is used to improve the training of deep networks.")
    elif selected_model == "Xception_model":
        st.write("Xception Modeli: Xception is a model that fundamentally changes the convolutional neural network architecture. It efficiently extracts features and can be used for classification tasks.")
    elif selected_model == "NASNetMobile_model":
        st.write("NASNetMobile Modeli: NASNetMobile is a model developed with automatic architecture search and optimized specifically for lightweight and mobile devices. It can be used for transfer learning for mobile applications and portable devices.")   
    elif selected_model == "EfficientNetB0_model":
        st.write("EfficientNetB0_model, It is the smallest model of the 'EfficientNet' family and carefully optimizes the network structure. It is suitable for visual processing tasks that require high performance and low computational cost.")
           
# Tahmin yap butonu
if st.button("Guess"):
    if upload_method == "Download from your computer" and uploaded_image is not None:
        image = Image.open(uploaded_image)
    elif upload_method == "Install with Internet Connection" and image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
        except Exception as e:
            st.error("There was an error uploading the image. Please enter a valid internet link.")

    # Kullanıcının seçtiği modele göre modeli yükle
    if selected_model == "CNN_model":
        model_path = 'CNN_model.h5'
    elif selected_model == "VGG16_model":
        model_path = 'VGG16.h5'
    elif selected_model == "ResNet_model":
        model_path = 'Resnet50.h5'
    elif selected_model == "Xception_model":
        model_path = 'Xception.h5'
    elif selected_model == "NASNetMobile_model":
        model_path = 'NASNetMobile.h5'
    elif selected_model == "EfficientNetB0_model":
        model_path = 'EfficientNetV2_S_model_trial_2.h5'
    elif selected_model == "mobilenet_model":
        model_path = 'mobilenet.h5'
    

    # Seçilen modeli yükle
    model = tf.keras.models.load_model(model_path, compile=False)   # , compile=False

    # Resmi model için hazırla ve tahmin yap
    if 'image' in locals():
      
        if model_path == 'cnn2-resnet50.h5':
            image = image.resize((224, 224))
            image = tf.keras.preprocessing.image.img_to_array(image)
            #image = image / 255.0
            image = np.expand_dims(image, axis=0)
        else:
            image = image.resize((224, 224))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = np.expand_dims(image, axis=0)

        # Tahmin yap
        prediction = model.predict(image)

        # Tahmin sonuçlarını göster
        class_names = ["Dandelion", "Daisy", "Sunflower" , "Tulip","Rose"]  # Modelin tahmin sınıfları
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction)
        
        st.markdown(f'<p style="background-color: #8FB447; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">🌷Model Prediction🌷</p>', unsafe_allow_html=True)

        st.write(f"Prediction Result: {predicted_class}")
        st.write(f"Prediction Confidence: {confidence:.2f}")
        
        st.markdown('<p style="background-color: #8FB447; color: white; font-size: 20px; padding: 10px; border-radius: 5px; text-align: center; box-shadow: 0px 2px 3px rgba(0, 0, 0, 0.1);">📊 Prediction Probabilities 📊</p>', unsafe_allow_html=True)
        prediction_df = pd.DataFrame({'Flower Types': class_names, 'Probabilities': prediction[0]})
        st.bar_chart(prediction_df.set_index('Flower Types'))
         
        
        
    st.markdown('<div style="display: flex; justify-content: flex-end; margin-top:-70px"><img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExNW1sb3UzcWYwdXg2dmIzcWJlOW40N3MzeGNuejgwNXpvNzhsdWd0cSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/iIKrdvt54McJa/giphy.gif" alt="GIF" width="100%" style="max-width: 200px; margin-right: 250px;"></div>', unsafe_allow_html=True)
 
