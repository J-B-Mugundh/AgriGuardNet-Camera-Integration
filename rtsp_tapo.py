import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response, jsonify
from PIL import Image
import io

# Flask app initialization
app = Flask(__name__)

# Camera credentials
username = ""  # Replace with your RTSP username
password = ""  # Replace with your RTSP password

# RTSP URL for the main stream (HD)
rtsp_url = f"rtsp://{username}:{password}@192.168.1.10/stream1"  # Tapo IP Camera

## rtsp_url = "rtsp://admin:admin@192.168.1.7:1935" # Mobile RTSP

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

# Define class names for pests
CLASS_NAMES_PEST = [
    'Citrus Canker', 'Colorado Potato Beetles', 'Fall Armyworms', 'Cabbage Loopers',
    'Spider Mites', 'Corn Borers', 'Brown Marmorated Stink Bugs', 'Corn Earworms',
    'Thrips', 'Western Corn Rootworms', 'Tomato Hornworms', 'Armyworms',
    'Africanized Honey Bees (Killer Bees)', 'Fruit Flies', 'Aphids'
]

# Function to predict plant disease
def model_prediction(image, model_type="plant"):
    """Predict plant disease or pest from an image."""
    try:
        # Preprocess the image
        image = image.resize((224, 224))
        input_arr = np.array(image)
        input_arr = np.expand_dims(input_arr, axis=0).astype(np.float32) / 255.0

        if model_type == "plant":
            # Set the plant disease model input tensor
            disease_interpreter.set_tensor(disease_input_details[0]['index'], input_arr)
            disease_interpreter.invoke()
            output_data = disease_interpreter.get_tensor(disease_output_details[0]['index'])
            result_index = int(np.argmax(output_data))
            return CLASS_NAMES_PLANT[result_index]
        
        elif model_type == "pest":
            # Set the pest detection model input tensor
            pest_interpreter.set_tensor(pest_input_details[0]['index'], input_arr)
            pest_interpreter.invoke()
            output_data = pest_interpreter.get_tensor(pest_output_details[0]['index'])
            result_index = int(np.argmax(output_data))
            return CLASS_NAMES_PEST[result_index]
        
    except Exception as e:
        print(f"Error in model_prediction: {e}")
        return None

# Generator function to process and display RTSP video stream
def generate_processed_frames():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception("Could not open RTSP stream. Check the URL or network connection.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Define the detection region (center of the frame)
        height, width, _ = frame.shape
        box_size = min(height, width) // 4
        x_center, y_center = width // 2, height // 2
        x1, y1 = x_center - box_size, y_center - box_size
        x2, y2 = x_center + box_size, y_center + box_size

        # Extract the detection region for plant disease and pest
        detection_region = frame[y1:y2, x1:x2]

        # Predict disease in the detection region
        plant_disease_result = model_prediction(detection_region, model_type="plant")
        pest_result = model_prediction(detection_region, model_type="pest")

        # Draw the bounding box for plant disease and pest detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Display the plant disease prediction on top of the box
        plant_disease_text = f"Plant Disease: {plant_disease_result}"
        font_scale = 0.6
        thickness = 2
        text_size, _ = cv2.getTextSize(plant_disease_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_width, text_height = text_size

        text_x = max(x1, 0)
        text_y = max(y1 - 10, text_height + 10)  # Position above the box
        cv2.putText(frame, plant_disease_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), thickness, lineType=cv2.LINE_AA)

        # Display the pest detection prediction on top of the box
        pest_text = f"Pest: {pest_result}"
        cv2.putText(frame, pest_text, (text_x, text_y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 255), thickness, lineType=cv2.LINE_AA)

        # Encode frame to JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield the processed frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

# Flask Routes
@app.route('/video_feed')
def video_feed():
    """Route to stream video."""
    return Response(generate_processed_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    """Landing page."""
    return '''
    <html>
        <head>
            <title>Plant Disease and Pest Detection</title>
        </head>
        <body>
            <h1>Plant Disease and Pest Detection</h1>
            <p>View RTSP stream:</p>
            <img src="/video_feed" width="640" height="480">
            <p>This stream detects both plant diseases and pests. Detection is shown as bounding boxes on the stream.</p>
        </body>
    </html>
    '''

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
