import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
)

# Initialize drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (MediaPipe uses RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(frame_rgb)

    # Draw the pose landmarks on the frame
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS
        )

    # Show the frame with pose landmarks
    cv2.imshow('Pose Estimation', frame)

    #Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Realese resources
cap.release()
cv2.destroyAllWindows()