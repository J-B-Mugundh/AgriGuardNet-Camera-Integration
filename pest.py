import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image

# Flask app initialization
app = Flask(__name__)

# Load TFLite Model and allocate tensors for pest detection
pest_interpreter = tf.lite.Interpreter(model_path="pest_detection_model.tflite")
pest_interpreter.allocate_tensors()

# Get input and output tensors for the model
pest_input_details = pest_interpreter.get_input_details()
pest_output_details = pest_interpreter.get_output_details()

# Define class names for pests
CLASS_NAMES_PEST = [
    'Citrus Canker', 'Colorado Potato Beetles', 'Fall Armyworms', 'Cabbage Loopers',
    'Spider Mites', 'Corn Borers', 'Brown Marmorated Stink Bugs', 'Corn Earworms',
    'Thrips', 'Western Corn Rootworms', 'Tomato Hornworms', 'Armyworms',
    'Africanized Honey Bees (Killer Bees)', 'Fruit Flies', 'Aphids'
]

# Function to predict pests
def predict_pest(image):
    try:
        # Preprocess the image
        image = image.resize((224, 224))
        input_arr = np.array(image, dtype=np.float32) / 255.0
        input_arr = np.expand_dims(input_arr, axis=0)

        # Perform prediction
        pest_interpreter.set_tensor(pest_input_details[0]['index'], input_arr)
        pest_interpreter.invoke()
        output_data = pest_interpreter.get_tensor(pest_output_details[0]['index'])
        result_index = int(np.argmax(output_data))
        return CLASS_NAMES_PEST[result_index]
    except Exception as e:
        print(f"Error in pest prediction: {e}")
        return "Error in processing the image"

# Route for uploading an image
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        try:
            # Open the uploaded image
            image = Image.open(file.stream).convert('RGB')
            prediction = predict_pest(image)
            return jsonify({'prediction': prediction})
        except Exception as e:
            return jsonify({'error': f'Error processing file: {e}'})
    
    # Render the upload page
    return '''
    <!doctype html>
    <html>
        <head>
            <title>Pest Detection - Upload</title>
        </head>
        <body>
            <h1>Upload an Image for Pest Detection</h1>
            <form action="/upload" method="POST" enctype="multipart/form-data">
                <input type="file" name="file" accept="image/*">
                <button type="submit">Upload</button>
            </form>
        </body>
    </html>
    '''

# Flask Routes
@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h1>Pest Detection</h1>
            <p><a href="/upload">Upload an Image</a></p>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
