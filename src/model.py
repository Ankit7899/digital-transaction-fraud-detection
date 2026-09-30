import joblib
import os

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "models",
    "fraud_detection_rf.pkl"
)

model = joblib.load(MODEL_PATH)

print("Fraud detection model loaded successfully!")