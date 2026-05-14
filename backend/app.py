import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from model_utils import load_model, preprocess_image, predict_disease, get_recommendations

app = Flask(__name__)
CORS(app)

# Load the model once when the backend starts
model = load_model()

@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    image_file = request.files['image']
    sensors_raw = request.form.get('sensors', '{}')
    location_raw = request.form.get('location', '{}')

    try:
        sensors = json.loads(sensors_raw)
    except Exception:
        sensors = {}

    try:
        location = json.loads(location_raw)
    except Exception:
        location = {}

    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image_array = preprocess_image(image)

    disease, confidence = predict_disease(model, image_array)
    recommendations = get_recommendations(disease)

    response = {
        'disease': disease,
        'confidence': float(confidence),
        'recommendations': recommendations,
        'sensors': sensors,
        'location': location,
    }
    return jsonify(response)

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'project': 'HarvestShield'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
