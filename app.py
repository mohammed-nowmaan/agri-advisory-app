import joblib
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'models', 'crop_recommendation_model.pkl')
LE_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'models', 'label_encoder.pkl')

def _safe_load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    # Try joblib first (works for joblib and many pickle files)
    try:
        return joblib.load(path)
    except Exception:
        # Fallback to pickle
        with open(path, 'rb') as f:
            return pickle.load(f)


# Load trained model
model = _safe_load_model(MODEL_PATH)

# Load LabelEncoder (try joblib then pickle)
le = _safe_load_model(LE_PATH)

# Example input: [N, P, K, temperature, humidity, ph, rainfall]
input_data = [[90, 42, 43, 28, 75, 6.5, 200]]

# Predict
predicted_label = model.predict(input_data)

# Decode numeric label to crop name
predicted_crop = le.inverse_transform(predicted_label)
print(f"Recommended Crop: {predicted_crop[0]}")
