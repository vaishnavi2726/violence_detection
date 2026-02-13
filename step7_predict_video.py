import os
import numpy as np
import joblib
from step5_feature_extraction import extract_motion_features

# Load model and scaler
model = joblib.load("violence_model.pkl")
scaler = joblib.load("scaler.pkl")

folders = [("non_violent", 0), ("violent", 1)]

total = 0
correct = 0

print("\n==============================")
print("TESTING ALL VIDEOS")
print("==============================\n")

for folder, true_label in folders:
    path = os.path.join("data", folder)

    for video in os.listdir(path):
        video_path = os.path.join(path, video)

        print("Checking:", video_path)

        features = extract_motion_features(video_path)

        if features is None:
            print("❌ Could not read:", video_path)
            continue

        features = np.array([features])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        total += 1

        if prediction == true_label:
            correct += 1
            print("✅ Correct")
        else:
            print("❌ Wrong Prediction")

        print("-----------------------")

print("\n==============================")
print("FINAL RESULTS")
print("==============================")
print("Total Tested:", total)
print("Correct Predictions:", correct)
print("Accuracy:", round((correct / total) * 100, 2), "%")
