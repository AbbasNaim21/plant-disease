from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app) 
# Load the model
model = tf.keras.models.load_model('./trained_plant_disease_model.keras')

# Prediction function
def model_prediction(test_image):
    image = Image.open(io.BytesIO(test_image))
    image = image.resize((128, 128))  # Resize to match model input size
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Add batch dimension
    prediction = model.predict(input_arr)
    result = np.argmax(prediction)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read the image file
    img = file.read()
    result_index = model_prediction(img)
    
    # Reading Labels
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
    
    return jsonify({'prediction': class_name[result_index]})

if __name__ == '__main__':
    app.run(debug=True)
