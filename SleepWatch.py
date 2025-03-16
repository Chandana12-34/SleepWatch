import cv2
import numpy as np
import dlib
from imutils import face_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load the dlib face detector and shape predictor for facial landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize variables for tracking sleep, drowsy, and active states
sleep = 0
drowsy = 0
active = 0
status = ""
color = (0, 0, 0)

# Function to compute the Euclidean distance between two points
def compute(ptA, ptB):
    return np.linalg.norm(ptA - ptB)

# Function to check if the eyes are blinking based on landmarks
def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)  # Vertical eye distance
    down = compute(a, f)  # Horizontal eye distance
    ratio = up / (2.0 * down)
    if ratio > 0.25:
        return 2  # Blinked
    elif 0.21 <= ratio <= 0.25:
        return 1  # Drowsy (semi-open eyes)
    else:
        return 0  # Eyes open

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    # Initialize face_frame (in case no faces are detected)
    face_frame = frame.copy()

    # Process each face detected
    for face in faces:
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

        # Draw a rectangle around the face
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Get facial landmarks
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Detect blinking for both left and right eyes
        left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])

        # Update the sleep, drowsy, and active counters based on the blinking state
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            if sleep > 6:
                status = "SLEEPING !!!"
                color = (255, 0, 0)  # Red for sleeping
        elif left_blink == 1 or right_blink == 1:
            sleep = 0
            active = 0
            drowsy += 1
            if drowsy > 6:
                status = "Drowsy !"
                color = (0, 0, 255)  # Blue for drowsy
        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)  # Green for active

        # Display the status on the frame
        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # Draw the facial landmarks on the face
        for (x, y) in landmarks:
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    # Show the result
    cv2.imshow("Frame", frame)
    cv2.imshow("Result of detector", face_frame)

    # Exit the loop when the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
