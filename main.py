
import cv2
import numpy as np
import dlib
from imutils import face_utils
import csv
import time
import winsound  # Sound alert for drowsy or sleeping

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the face detector and the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# State variables
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)
last_sleep_time = 0
last_drowsy_time = 0
alarm_duration = 10  #when in "Drowsy" or "Sleeping" state, start the Alarm

# CSV file to log the results
csv_file = "biometric_test.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Screenshot", "Status", "Timestamp"])  #headers

# Function to compute Euclidean distance
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to detect blink
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2  # Eye open
    elif 0.21 < ratio <= 0.25:
        return 1  # Drowsy
    else:
        return 0  # Eye closed

def save_to_csv(image_path, status, timestamp):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_path, status, timestamp])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    face_frame = frame.copy()

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Blink detection
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38],
                             landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44],
                              landmarks[47], landmarks[46], landmarks[45])

        # Sleep, drowsy, or active states based on blink detection
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)
                current_time = time.time()
                if current_time - last_sleep_time >= alarm_duration:
                    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                    csv_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    screenshot_path = f"screenshots/sleeping_{timestamp}.png"
                    cv2.imwrite(screenshot_path, frame)
                    save_to_csv(screenshot_path, status, csv_timestamp)
                    winsound.Beep(1000, 1000)
                    last_sleep_time = current_time

        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)
                current_time = time.time()
                if current_time - last_drowsy_time >= alarm_duration:
                    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                    csv_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                    screenshot_path = f"screenshots/drowsy_{timestamp}.png"
                    cv2.imwrite(screenshot_path, frame)
                    save_to_csv(screenshot_path, status, csv_timestamp)
                    winsound.Beep(500, 1000)
                    last_drowsy_time = current_time

        else:
            drowsy = 0
            sleep = 0
            active += 1
            status = "Active :)"
            color = (0, 255, 0)

        # Display the status on the frame
        cv2.putText(frame, status, (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw landmarks on the face
        for (x, y) in landmarks:
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

        # Draw rectangle around face
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the frames
    cv2.imshow("Frame", frame)
    cv2.imshow("Landmarks", face_frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
