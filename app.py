import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import io

# Flask app initialization
app = Flask(__name__)

# Load TFLite Model and allocate tensors for plant disease detection
disease_interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
disease_interpreter.allocate_tensors()

# Get input and output tensors for the model
disease_input_details = disease_interpreter.get_input_details()
disease_output_details = disease_interpreter.get_output_details()

# Define class names for plant diseases
CLASS_NAMES_PLANT = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
    'Tomato___healthy'
]

# Function to predict plant disease
def predict_plant_disease(image):
    try:
        # Preprocess the image
        image = image.resize((224, 224))
        input_arr = np.array(image, dtype=np.float32) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)

        # Perform prediction
        disease_interpreter.set_tensor(disease_input_details[0]['index'], input_arr)
        disease_interpreter.invoke()
        output_data = disease_interpreter.get_tensor(disease_output_details[0]['index'])
        result_index = int(np.argmax(output_data))
        return CLASS_NAMES_PLANT[result_index]
    except Exception as e:
        print(f"Error in plant disease prediction: {e}")
        return None

# Flask Routes
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <title>Plant Disease Detection</title>
    </head>
    <body>
        <h1>Plant Disease Detection</h1>
        <form action="/predict" method="POST" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Upload and Predict</button>
        </form>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    try:
        # Open the image file
        image = Image.open(io.BytesIO(file.read())).convert('RGB')

        # Predict the plant disease
        prediction = predict_plant_disease(image)

        if prediction:
            return f'''
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <title>Prediction Result</title>
            </head>
            <body>
                <h1>Prediction Result</h1>
                <p>Plant Disease: {prediction}</p>
                <a href="/">Back to Home</a>
            </body>
            </html>
            '''
        else:
            return jsonify({"error": "Prediction failed."}), 500

    except Exception as e:
        return jsonify({"error": f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
