import os
import random
import numpy as np
from PIL import Image

# Force Keras 3 to use TensorFlow backend
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'harvest_model.h5')

def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            print(f"Keras version: {keras.__version__}")
            print(f"Loading model from {MODEL_PATH}...")
            model = keras.models.load_model(MODEL_PATH, compile=False)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    else:
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)

def predict_disease(model, image_array):
    if model is None:
        raise ValueError("Model is not loaded.")
        
    CLASS_LABELS = [
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_healthy"
    ]
    
    result = model.predict(image_array)
    probabilities = result[0]  # Full softmax vector for all classes
    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    
    # Scale confidence to 0-100 for the fusion engine
    confidence_100 = round(confidence * 100, 2)
    
    # Return full probability vector for downstream calibration
    full_probs = [float(p) for p in probabilities]
    
    return CLASS_LABELS[index], confidence_100, full_probs


def predict_image(model, image_array):
    """Wrapper that returns results as a dictionary for the orchestrator."""
    label, conf, probs = predict_disease(model, image_array)
    return {
        'prediction': label,
        'confidence_score': conf,
        'full_probabilities': probs
    }

