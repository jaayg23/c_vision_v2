# pip install ultralytics

import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load the model YOLOv11 to segment images
model = YOLO("yolov11n-seg.pt")

# Define the video source(0 for webcam, or a video file path)
cap  = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Start the timer
    start_time = time.time()

    # Perform segmentation
    results = model(
        frame,
        conf=0.7, # Confidence threshold
        classes=[0] # Specify the classes to detect, e.g., [0] for person, [1] for bicycle, etc.
        )

    latency = (time.time()- start_time) * 1000  # Convert to milliseconds

    #Acces to detections
    boxes_obj = results[0].boxes
    if boxes_obj is not None and len(boxes_obj) > 0:
        bboxes = boxes_obj.xyxy.cpu().numpy()  # Get bounding boxes [x1, y1, x2, y2]
        confs = boxes_obj.conf.cpu().numpy()  # Get confidence scores
        classes = boxes_obj.cls.cpu().numpy()  # Get class indices

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)
            # Obtain the class name
            class_name = model.names[int(classes[i])] if hasattr(model, 'names') else str(int(classes[i]))
            label = f"{class_name} {confs[i]:.2f}"
            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    #Process the segmentations: assign colors to each detected mask
    masks_obj = results[0].masks
    if masks_obj is not None and len(masks_obj) > 0:
        #Extract masks; assume that masks_obj.data is a tensor:
        masks = masks_obj.data.cpu().numpy()  if hasattr(masks_obj, 'cpu') else masks_obj.numpy()
        for mask in masks:
            #Convert mask to binary (0.5 threshold) and scaling to 0-255
            mask_bin = (mask > 0.5).astype(np.uint8) * 255
            #Resize mask to match the frame size
            mask_bin = cv2.resize(mask_bin, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

            #Create a boolean mask 3 channels
            binary_mask = cv2.threshold(mask_bin, 127, 255, cv2.THRESH_BINARY)[1]
            binary_mask_3c = cv2.merge([binary_mask, binary_mask, binary_mask])

            #Generate a random color for the mask
            random_color = np.random.randint(0, 256, size=(3,), dtype=np.uint8)
            #Create an image same size as the frame
            colored_mask = np.full((frame.shape[0], frame_shape[1], 3), dtype=np.uint8)

            #Combine the mask with the frame: in the regions where the mask is 255, apply the color
            output_frame = frame.copy()
            output_frame[binary_mask_3c == 255] = colored_mask[binary_mask_3c == 255]

            #Update the frame with the colored mask
            frame = output_frame

        #Show the quantity of masks detected
        cv2.putText(frame, f"Masks: {len(masks)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    #Show the latency un frame
    cv2.putText(frame, f"Latency: {latency:.2f} ms", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
    #Show the fram processing time
    cv2.imshow("YOLOV11-Segmentation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()