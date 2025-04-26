from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
import os

app = Flask(__name__)

# Load your trained model
model = load_model('Model/deepfakedetector.h5')

# Function to predict
def detect_deepfake(img_path, model):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]
    result = "Fake" if prediction < 0.5 else "Real"
    confidence = (1 - prediction) if prediction < 0.5 else prediction

    return result, float(confidence)

# Route to upload image
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    result, confidence = detect_deepfake(filepath, model)

    os.remove(filepath)  # clean up after prediction

    return jsonify({'prediction': result, 'confidence': confidence})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
