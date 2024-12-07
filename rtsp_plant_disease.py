import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response
from PIL import Image

# Flask app initialization
app = Flask(__name__)

# Camera credentials
username = "mugundhjb@gmail.com"  # Replace with your RTSP username
password = "JBMK656040"  # Replace with your RTSP password

# RTSP URL for the main stream (HD)
# rtsp_url = f"rtsp://{username}:{password}@192.168.1.10/stream1"  # Tapo IP Camera

# Camera credentials
rtsp_url = "rtsp://admin:admin@192.168.1.7:1935"  # RTSP URL

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

# Video stream processing
def generate_frames():
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        raise Exception("Could not open RTSP stream. Check the URL or network connection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract detection region (center of the frame)
        height, width, _ = frame.shape
        size = min(height, width) // 4
        x1, y1, x2, y2 = (width // 2 - size, height // 2 - size, width // 2 + size, height // 2 + size)
        detection_region = frame[y1:y2, x1:x2]

        # Convert to PIL image and predict
        rgb_image = cv2.cvtColor(detection_region, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        prediction = predict_plant_disease(pil_image)

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Plant Disease: {prediction}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # Encode and yield frame
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

# Flask Routes
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <body>
            <h1>Plant Disease Detection</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
