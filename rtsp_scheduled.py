import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, jsonify
from PIL import Image
import io
import os

# Define folder to save captured images
CAPTURE_FOLDER = "captures"

# Ensure the folder exists
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# Flask app initialization
app = Flask(__name__)

# Camera credentials
username = "mugundhjb@gmail.com"  # Replace with your RTSP username
password = "JBMK656040"  # Replace with your RTSP password

# rtsp_url = f"rtsp://{username}:{password}@192.168.1.10/stream1"  # Tapo IP Camera

# RTSP URL for the main stream (HD)
rtsp_url = "rtsp://admin:admin@192.168.1.7:1935"  # Replace with your camera/mobile RTSP

# Load TFLite Models and allocate tensors for plant disease and pest detection
disease_interpreter = tf.lite.Interpreter(model_path="plant_disease_model.tflite")
disease_interpreter.allocate_tensors()

pest_interpreter = tf.lite.Interpreter(model_path="pest_detection_model.tflite")
pest_interpreter.allocate_tensors()

# Get input and output tensors for both models
disease_input_details = disease_interpreter.get_input_details()
disease_output_details = disease_interpreter.get_output_details()

pest_input_details = pest_interpreter.get_input_details()
pest_output_details = pest_interpreter.get_output_details()

# Define class names for plant diseases
CLASS_NAMES_PLANT = [
    'Scab', 'Black_rot', 'Rust', 'healthy',
    'healthy', 'downy_mildew ',
    'healthy', 'Cercospora_leaf_spot Gray_leaf_spot',
    'Common_rust_', 'Northern_Leaf_Blight', 'healthy',
    'Black_rot', 'Esca_Black_Measles', 'Leaf_blight_(Isariopsis_Leaf_Spot)',
    'healthy', 'Haunglongbing_(Citrus_greening)', 'Bacterial_spot',
    'healthy', 'Bacterial_spot', 'healthy',
    'Early_blight', 'Late_blight', 'healthy',
    'healthy', 'healthy', 'Powdery_mildew',
    'Leaf_scorch', 'healthy', 'Bacterial_spot',
    'Early_blight', 'Late_blight', 'Leaf_Mold',
    'Septoria_leaf_spot', 'Spider_mites Two-spotted_spider_mite',
    'Target_Spot', 'Yellow_Leaf_Curl_Virus', 'mosaic_virus',
    'healthy'
]

# Define class names for pests
CLASS_NAMES_PEST = [
    'Citrus Canker', 'Colorado Beetles', 'Fall Armyworms', 'Loopers',
    'Spider Mites', 'Borers', 'Brown Marmorated Stink Bugs', 'Earworms',
    'Thrips', 'Western Rootworms', 'Hornworms', 'Armyworms',
    'Honey Bees', 'Fruit Flies', 'Aphids'
]

# Function to predict plant disease or pest
def model_prediction(image, model_type="plant"):
    try:
        # Preprocess the image
        image = image.resize((224, 224))  # Resize image to fit model input size
        input_arr = np.array(image)  # Convert image to numpy array
        input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32)  # Add batch dimension and convert to float32
        input_arr = input_arr / 255.0  # Normalize image

        if model_type == "plant":
            # Set tensor for the plant disease model
            disease_interpreter.set_tensor(disease_input_details[0]['index'], input_arr)
            disease_interpreter.invoke()  # Invoke the interpreter to run inference
            output_data = disease_interpreter.get_tensor(disease_output_details[0]['index'])  # Get the output tensor
            result_index = int(np.argmax(output_data))  # Get the class index with the highest probability
            return CLASS_NAMES_PLANT[result_index]  # Return the class name for plant disease

        elif model_type == "pest":
            # Set tensor for the pest detection model
            pest_interpreter.set_tensor(pest_input_details[0]['index'], input_arr)
            pest_interpreter.invoke()  # Invoke the interpreter to run inference
            output_data = pest_interpreter.get_tensor(pest_output_details[0]['index'])  # Get the output tensor
            result_index = int(np.argmax(output_data))  # Get the class index with the highest probability
            return CLASS_NAMES_PEST[result_index]  # Return the class name for pest

    except Exception as e:
        print(f"Error in model_prediction: {e}")
        return None

# Route to stream live video feed with overlayed results
def generate_frames():
    cap = cv2.VideoCapture(rtsp_url)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Convert the captured frame to a PIL image for prediction
            detection_region_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(detection_region_rgb)

            # Predict plant disease and pest
            plant_disease_result = model_prediction(pil_image, model_type="plant")
            pest_result = model_prediction(pil_image, model_type="pest")

            # Overlay the predictions on the frame
            cv2.putText(frame, f"Plant Disease: {plant_disease_result}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Pest: {pest_result}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            # Yield frame data for MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to capture an image and make predictions
@app.route('/capture', methods=['POST'])
def capture_frame():
    try:
        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open RTSP stream. Check the URL or network connection.'})

        success, frame = cap.read()
        cap.release()

        if not success:
            return jsonify({'error': 'Failed to capture image from the stream.'})

        # Convert BGR to RGB
        detection_region_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(detection_region_rgb)

        # Predict plant disease and pest
        plant_disease_result = model_prediction(pil_image, model_type="plant")
        pest_result = model_prediction(pil_image, model_type="pest")

        # Save the captured image to the folder
        filename = os.path.join(CAPTURE_FOLDER, "capture.jpg")
        cv2.imwrite(filename, frame)

        return jsonify({
            'message': 'Image captured successfully.',
            'plant_disease': plant_disease_result,
            'pest': pest_result,
            'saved_path': filename
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# Updated landing page
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Plant Disease and Pest Detection</title>
        <script>
            function captureImage() {
                fetch('/capture', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        alert("Error: " + data.error);
                    } else {
                        alert(
                            `Capture successful!\\nPlant Disease: ${data.plant_disease}\\nPest: ${data.pest}\\nSaved Path: ${data.saved_path}`
                        );
                    }
                })
                .catch(error => alert("Error: " + error));
            }
        </script>
    </head>
    <body>
        <div style="text-align: center; margin-top: 50px;">
            <h1>Plant Disease and Pest Detection</h1>
            <p>Click the button below to capture a frame from the live stream and detect plant diseases and pests.</p>
            <button onclick="captureImage()">Capture Image and Detect</button>
            <div style="margin-top: 20px;">
                <h2>Live Stream</h2>
                <img src="/video_feed" alt="Live stream">
            </div>
        </div>
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
