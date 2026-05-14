import os
import random
import numpy as np
from PIL import Image

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'harvest_model.h5')

# Placeholder loader; replace with your actual TensorFlow/Keras model load
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception:
            pass
    return None

# Placeholder preprocessing; adapt to your real CNN requirements
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    array = np.array(image) / 255.0
    return np.expand_dims(array, axis=0)

# Placeholder prediction; replace with real model inference
def predict_disease(model, image_array):
    diseases = ['Healthy', 'Paddy Leaf Blast', 'Cotton Boll Rot', 'Tomato Early Blight']
    if model is not None:
        try:
            result = model.predict(image_array)
            index = int(np.argmax(result, axis=1)[0])
            confidence = float(np.max(result, axis=1)[0])
            return diseases[index], confidence
        except Exception:
            pass

    # fallback mock prediction
    disease = random.choice(diseases)
    return disease, 0.78

def get_recommendations(disease):
    mapping = {
        'Healthy': [
            'Crop is healthy. Maintain current care plan.',
            'Monitor soil moisture and temperature regularly.',
        ],
        'Paddy Leaf Blast': [
            'Apply Tricyclazole-based fungicide.',
            'Avoid heavy nitrogen fertilization until infection subsides.',
            'Ensure good drainage in the field.',
        ],
        'Cotton Boll Rot': [
            'Remove infected bolls and dispose of them safely.',
            'Improve airflow by spacing plants properly.',
        ],
        'Tomato Early Blight': [
            'Use Chlorothalonil or copper-based sprays.',
            'Remove lower leaves to reduce soil splash.',
        ],
    }
    return mapping.get(disease, ['No recommendation available.'])
