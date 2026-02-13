🛑 Violence Detection System

A Machine Learning–based system for detecting violent activity in video clips using motion analysis and feature extraction techniques.

📌 Overview

This project detects violent scenes in videos by analyzing motion patterns between frames and training a classifier on extracted features.

The system follows this pipeline:

📹 Video Reading

🎥 Motion Detection

📊 Feature Extraction

🤖 Model Training

🔎 Violence Prediction






🗂️ Project Structure
violence_detection/
│
├── step2_video_read.py
├── step4_motion_detection.py
├── step5_feature_extraction.py
├── step6_train_model.py
├── step7_predict_video.py
├── scaler.pkl
├── violence_model.pkl
├── README.md
└── .gitignore







⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/vaishnavi2726/violence_detection.git
cd violence_detection

2️⃣ Install Dependencies

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate   # Mac/Linux


Install required libraries:

pip install opencv-python numpy scikit-learn
