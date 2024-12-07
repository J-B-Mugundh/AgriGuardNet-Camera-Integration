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
        prediction = predict_pest(pil_image)

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Pest: {prediction}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
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
            <h1>Pest Detection</h1>
            <img src="/video_feed" width="640" height="480">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
