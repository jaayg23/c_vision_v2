import cv2
import mediapipe as mp
import numpy as np

def add_gaussian_to_heatmap(heatmap, center, sigma=15, amplitude=50):
    """
    Add a Gaussian blob to the heatmap at the specified center.

    Parameters:
        heatmap (np.ndarray): Activation heatmap (h x w) which represent the distribution of looking
        center (tuple): The (cx, cy) coordinates of the center of the Gaussian blob.
        sigma (int): The standard deviation of the Gaussian distribution.
        amplitute (float): Maximum value of the intensity to sum.

    returns:
        np.ndarray: Updated heatmap with the Gaussian blob added.
    """
    h, w = heatmap.shape
    y, x = np.indices((h,w))
    cx, cy = center

    #Calculate the Gaussian distribution
    gaussian = amplitude * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma ** 2))

    heatmap += gaussian #Sum the Gaussian to the heatmap
    return heatmap

#Parametres of configuration

#decay factor for the heatmap, allowing that old activations fade over time
# Values near 1 makes the trace of lookin more persistent
decay_factor = 0.98

# Disperse size of the Gaussian blob
sigma = 15

#Maximun intesnity each blob can have
amplitude = 50

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

#Configurate the window to show in entire screen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

ret,frame = cap.read()
h,w,_ = frame.shape

# Initialize the heatmap with zeros
heatmap_gaze = np.zeros((h, w), dtype=np.float32)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #Apply the decay factor to the heatmap
    heatmap_gaze *= decay_factor

    # Convert the frame to RGB (MediaPipe uses RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Acquire key points in eyes
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]

            # Get the absolute coordinates of the eyes
            left_eye_coords = (int(left_eye.x * w), int(left_eye.y * h))
            right_eye_coords = (int(right_eye.x * w), int(right_eye.y * h))

            # Calculate the median point between the eyes
            mid_eye = ((left_eye_coords[0] + right_eye_coords[0]) // 2,
                         (left_eye_coords[1] + right_eye_coords[1]) // 2)

            # Add a Gaussian blob to the heatmap at the median eye position
            heatmap_gaze = add_gaussian_to_heatmap(heatmap_gaze, mid_eye, sigma, amplitude)

    # Normalize the heatmap for visualization
    heatmap_vis = cv2.normalize(heatmap_gaze, None, 0, 255, cv2.NORM_MINMAX)
    heatmap_vis = np.uint8(heatmap_vis)  # Convert to uint8 for visualization
    heatmap_vis = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)

    # Overlay the heatmap on the original frame
    overlay = cv2.addWeighted(frame, 0.6, heatmap_vis, 0.4, 0)

    # Show the frame with the heatmap overlay
    cv2.imshow('Gaze Heatmap', overlay)
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()  # Close the MediaPipe Face Mesh instance