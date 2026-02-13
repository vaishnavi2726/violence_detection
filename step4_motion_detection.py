import cv2
import os

folder = "data/violent"

for filename in os.listdir(folder):
    if filename.endswith(".mp4"):

        path = os.path.join(folder, filename)
        print("Processing:", filename)

        cap = cv2.VideoCapture(path)

        ret, prev_frame = cap.read()
        if not ret:
            print("Error reading:", filename)
            continue

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

            motion_score = thresh.sum()

            print("Motion score:", motion_score)

            if motion_score > 500000:
                cv2.putText(frame, "HIGH MOTION",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2)

            cv2.imshow("Motion Detection", frame)

            prev_gray = gray

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()

cv2.destroyAllWindows()
