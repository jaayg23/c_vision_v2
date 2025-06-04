import cv2
import mediapipe as mp

""""
Capture video from the webcam, process each frame with MediaPipe Face Mesh
(with refined landmarks), and draw:
    - Eyes landmarks (green)
    - 4 landmarks used to calculate the iris center (red)
    - Iris center (blue)
"""

# Initialize MediaPipe Face Mesh with refined landmarks (for iris detection)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,  # Enable refined landmarks for iris detection
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Acquire dimensions of the frame
            h, w, _ = frame.shape

            # ------------------------------------------
            # Detect eyes landmarks (33 and 263 indices)
            # ------------------------------------------
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

            # Draw the eyes landmarks
            cv2.circle(frame, left_eye_coords, 3, (0, 255, 0), -1)
            cv2.circle(frame, right_eye_coords, 3, (0, 255, 0), -1)

            # --------------------------------------------------
            # Detect 4 landmarks used to calculate the iris center
            # --------------------------------------------------
            # Left eye, indices 468, 469, 470, 471
            left_iris_points = []
            for i in range(468, 468 + 4):
                pt = face_landmarks.landmark[i]
                x , y = int(pt.x * w), int(pt.y * h)
                left_iris_points.append((x, y))
                # Draw the left iris points
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # Calculate the left iris center
            left_iris_center = (
                int(sum(p[0] for p in left_iris_points) / len(left_iris_points)),
                int(sum(p[1] for p in left_iris_points) / len(left_iris_points))
            )

            # Right eye, indices 473, 474, 475, 476
            right_iris_points = []
            for i in range(473, 473 + 4):
                pt = face_landmarks.landmark[i]
                x, y = int(pt.x * w), int(pt.y * h)
                right_iris_points.append((x, y))
                # Draw the right iris points
                cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)

            # Calculate the right iris center
            right_iris_center = (
                int(sum(p[0] for p in right_iris_points) / len(right_iris_points)),
                int(sum(p[1] for p in right_iris_points) / len(right_iris_points))
            )   

            # Draw the iris centers
            cv2.circle(frame, left_iris_center, 3, (255, 0, 0), -1)
            cv2.circle(frame, right_iris_center, 3, (255, 0, 0), -1)

    # Show the frame with face landmarks
    cv2.imshow('Face Mesh - Iris Detection', frame)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()  # Close the MediaPipe Face Mesh instance