import cv2
import os

# Path to your folder
violent_folder = '/Users/vaishnavisingh/violence_detection/data/violent'
non_violent_folder = '/Users/vaishnavisingh/violence_detection/data/non_violent'

def process_videos(folder):
    for filename in os.listdir(folder):
        if filename.endswith(('.mp4', '.mov')):  # Add other formats if needed
            path = os.path.join(folder, filename)
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"Cannot open {filename}")
                continue
            print(f"Opened {filename} successfully")
            
            # Example: read frames (optional)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Do something with frame, e.g., resize or save
                # frame = cv2.resize(frame, (224, 224))
            
            cap.release()

# Process violent and non-violent videos
process_videos(violent_folder)
process_videos(non_violent_folder)
