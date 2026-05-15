import io
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from model_utils import load_model, preprocess_image, predict_disease
from fusion_engine import advanced_fusion_engine
from climate_utils import get_climate_context
import tempfile
import os

app = Flask(__name__)
CORS(app)

# Load the model once when the backend starts
try:
    model = load_model()
except Exception as e:
    print(f"Warning: Failed to load model on startup: {e}")
    model = None

@app.route('/api/detect', methods=['POST'])
def detect():
    print("\n>>> RECEIVED DETECTION REQUEST")
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400
        
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    image_file = request.files['image']
    sensors_raw = request.form.get('sensors', '{}')
    location_raw = request.form.get('location', '{}')

    print(f"[DEBUG-APP] Raw Sensors: {sensors_raw}")
    print(f"[DEBUG-APP] Raw Location: {location_raw}")

    try:
        sensors = json.loads(sensors_raw)
    except Exception as e:
        print(f"[DEBUG-APP] Sensor JSON Parse Error: {e}")
        sensors = {}
        
    temperature = float(sensors.get('temperature', 25.0))
    humidity = float(sensors.get('humidity', 60.0))
    # Handle 'soil' (old backend), 'soilMoisture' (frontend), or 'soil_moisture' (standard)
    soil = float(sensors.get('soil', sensors.get('soilMoisture', sensors.get('soil_moisture', 50.0))))

    print(f"[DEBUG-APP] Extracted Sensors -> Temp: {temperature}, Hum: {humidity}, Soil: {soil}")

    try:
        loc_data = json.loads(location_raw)
        lat = loc_data.get('latitude')
        lon = loc_data.get('longitude')
    except Exception as e:
        print(f"[DEBUG-APP] Location JSON Parse Error: {e}")
        lat, lon = None, None

    # Get geo-climate context (Offline-first + Optional Online)
    climate_ctx = get_climate_context(lat, lon)

    try:
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Save temp file for OpenCV variance analysis in fusion_engine
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            temp_path = temp.name
            image.save(temp_path)
            
        try:
            # Initialize Hierarchical Orchestrator
            from reasoning_orchestrator import HierarchicalOrchestrator
            orchestrator = HierarchicalOrchestrator(model)
            
            # Prepare request data
            sensor_data = {
                "temperature": temperature,
                "humidity": humidity,
                "soil": soil
            }
            
            location = {
                "latitude": lat,
                "longitude": lon
            }
            
            # Run the full reasoning pipeline
            reasoning_result = orchestrator.process_request(
                image_path=temp_path,
                sensor_data=sensor_data,
                location=location
            )
            
            return jsonify(reasoning_result)

            
        finally:
            # Clean up the temporary image file
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:

        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'project': 'HarvestShield'})

if __name__ == '__main__':
    print(">>> STARTING FLASK SERVER ON PORT 5005")
    app.run(host='0.0.0.0', port=5005, debug=False)