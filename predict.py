import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "models", "kubernetes_issue_classifier.pkl")
scaler_path = os.path.join(base_dir, "models", "scaler.pkl")
encoder_path = os.path.join(base_dir, "models", "label_encoder.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
label_encoder = joblib.load(encoder_path)

def predict_issue(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    label = label_encoder.inverse_transform(prediction)
    return label[0]

if __name__ == "__main__":
    print("Enter the feature values separated by commas (e.g., 38.71,39.89,74.75,641.76,7,3,249.75,2.56,77):")
    user_input = input(">> ")
    try:
        features = list(map(float, user_input.strip().split(",")))
        result = predict_issue(features)
        print(f"\nPredicted Issue: {result}")
    except Exception as e:
        print(f"Error: {e}")
