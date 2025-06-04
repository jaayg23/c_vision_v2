import cv2
import mediapipe as mp

# Intialize Face Mesh 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the face landmarks
    results = face_mesh.process(frame_rgb)

    # Draw the face landmarks on the frame
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            # Get the absolute coordinates of the eyes
            h, w, _ = frame.shape
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

            #Calculate the median point between the eyes
            mid_eye_x = ((left_eye_coords[0] + right_eye_coords[0]) // 2,
                        (left_eye_coords[1] + right_eye_coords[1]) // 2)
            
            # Draw the eyes landmarks
            cv2.circle(frame, left_eye_coords, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_coords, 3, (0, 255, 0), -1)
            cv2.circle(frame, mid_eye_x, 3, (255, 0, 0), -1)

    # Show the frame with face landmarks
    cv2.imshow('Face Mesh', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()  # Close the MediaPipe Face Mesh instance