import os
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from step5_feature_extraction import extract_motion_features


features = []
labels = []

# (folder_name, label)
folders = [("non_violent", 0), ("violent", 1)]

print("Loading videos...\n")

for folder, label in folders:
    path = os.path.join("data", folder)

    for video in os.listdir(path):
        video_path = os.path.join(path, video)

        if not video.endswith(".mp4"):
            continue

        print("Processing:", video_path)

        values = extract_motion_features(video_path)

        if values is not None:
            features.append(values)
            labels.append(label)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

print("\nDataset Summary:")
print("Total samples:", len(y))
print("Non-violent:", list(y).count(0))
print("Violent:", list(y).count(1))


# Safety check
if len(set(y)) < 2:
    print("Error: Need at least one violent and one non-violent video.")
    exit()


# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# -----------------------------
# Evaluation
# -----------------------------
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("\nTraining Accuracy:", round(train_acc * 100, 2), "%")
print("Testing Accuracy:", round(test_acc * 100, 2), "%")


# -----------------------------
# Save Model
# -----------------------------
joblib.dump(model, "violence_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel and Scaler saved successfully!")
