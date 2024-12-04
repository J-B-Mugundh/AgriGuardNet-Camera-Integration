from flask import Flask, Response, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import io
from flask_cors import CORS

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# RTSP stream URL
rtsp_url = "rtsp://admin:admin@192.168.1.7:1935"

# Load TFLite Model and allocate tensors
interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class names
CLASS_NAMES = [
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

def model_prediction(image):
    """Predict plant disease from an image."""
    try:
        # Preprocess the image
        image = image.resize((224, 224))
        input_arr = np.array(image)
        input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32) / 255.0

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_arr)

        # Run the model
        interpreter.invoke()

        # Get the output tensor
        output_data = interpreter.get_tensor(output_details[0]['index'])
        return int(np.argmax(output_data))
    except Exception as e:
        print(f"Error in model_prediction: {e}")
        return None

def generate_frames():
    """Generator function to yield video frames."""
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception("Could not open RTSP stream. Check the URL or network connection.")

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    """Route to stream video."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    """Route for plant disease prediction."""
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No image selected for uploading"}), 400

    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        result_index = model_prediction(image)

        if result_index is None:
            return jsonify({"error": "Prediction failed"}), 500

        class_name = CLASS_NAMES[result_index]
        return jsonify({"prediction": class_name}), 200

    except Exception as e:
        print(f"Error in /predict: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

@app.route('/')
def index():
    """Landing page."""
    return '''
    <html>
        <head>
            <title>Plant Disease Detection</title>
        </head>
        <body>
            <h1>Plant Disease Detection</h1>
            <p>View RTSP stream:</p>
            <img src="/video_feed" width="640" height="480">
            <p>Use the /predict endpoint for plant disease detection by uploading an image.</p>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
