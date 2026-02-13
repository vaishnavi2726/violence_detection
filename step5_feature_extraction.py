import cv2
import numpy as np

def extract_motion_features(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        return None

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    motion_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        motion_values.append(np.sum(thresh))
        prev_gray = gray

    cap.release()

    if len(motion_values) == 0:
        return None

    return [np.mean(motion_values), np.max(motion_values)]
