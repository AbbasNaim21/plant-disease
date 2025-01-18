import streamlit as st
import tensorflow as tf 
import numpy as np


#Prediction 
def model_prediction(test_image):
    # Load the model
    model = tf.keras.models.load_model('./trained_plant_disease_model.keras')
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #converting to the batch format 
    prediction = model.predict(input_arr)
    result = np.argmax(prediction)
    return result


# Design 
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", {"Home", "About", "Disease Recognition"})

if(app_mode=='Home'):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "./bgg.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    # Welcome to the Plant Disease Detection Hub! 🌿🔬

Our platform is designed to assist you in detecting plant diseases quickly and accurately. Simply upload a photo of your plant, and we’ll analyze it to spot any signs of illness. Together, we can protect our crops and foster a healthier environment!

## How It Works

1. **Upload Your Image:**  
   Visit the **Disease Recognition** section and submit a photo of the plant you suspect is affected by disease.

2. **Image Analysis:**  
   Our system uses sophisticated algorithms to assess the image and identify any diseases.

3. **View Results:**  
   Check the diagnosis and recommended steps for treatment or action.

## Why Us?

- **Precision:**  
  We apply the latest advancements in machine learning to ensure highly accurate disease detection.

- **Simplicity:**  
  Our easy-to-use interface guarantees a smooth experience, no matter your tech skill.

- **Speed:**  
  Get results in a matter of seconds, enabling you to act swiftly.

## Start Now

Head to the **Disease Recognition** page from the sidebar, upload your image, and unlock the power of our disease detection system!

## Learn More About Us

Find out more about our mission, team, and vision by visiting the **About** page.

    """)


elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### Dataset Overview
                This dataset has been generated through offline augmentation techniques based on the original dataset, which can be accessed on the GitHub repository. 
                It contains approximately 87,000 RGB images of both healthy and infected crop leaves, spread across 38 distinct categories. 
                The dataset is split into an 80/20 ratio for training and validation, ensuring the folder structure remains intact.
                Additionally, a new directory with 33 test images has been added for prediction purposes.
                
                #### Breakdown of Dataset
                1. **Training Set**: 70,295 images
                2. **Test Set**: 33 images
                3. **Validation Set**: 17,572 images
                """)


elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))