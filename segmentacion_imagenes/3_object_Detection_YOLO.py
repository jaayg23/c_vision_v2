import cv2 
import time 
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n-seg.pt") 

# Open the video file
cap = cv2.VideoCapture()

# Configure the window
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #Measure the time taken for detection
    start_time = time.time()

    # Perform object detection
    results = model(
        frame, 
        # conf= 0.7
        #classes=[0]

    )

    #Calculate the latency
    latency = (time.time() - start_time) * 1000  # Convert to milliseconds

    #Acces the firs result (Yolov11 returns a list of results)
    boxes_obj = results[0].boxes

    #if detection is not empty, extract information and draw boxes
    if boxes_obj is not None and len(boxes_obj) > 0:
        #Extract the bounding boxes, confidences and class ids
        bboxes = boxes_obj.xyxy.cpu().numpy()  # Bounding boxes
        confs = boxes_obj.conf.cpu().numpy()  # Confidence scores
        classes = boxes_obj.cls.cpu().numpy() # Class IDs

        #Iterate over the detected objects and draw bounding boxes
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, box)

            # Acquire the class name; if the model hasn't 'names', use the class ID
            class_name = model.names[int(classes[i])] if hasattr(model, 'names') else str(int(classes[i]))

            # Create the label with class name and confidence
            label = f"{class_name} {confs[i]:.2f}"

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    #Show the latency on the frame to measure the performance
    cv2.putText(frame, f'Latency: {latency:.1f}ms', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Show the frame with detections
    cv2.imshow("YOLO Object Detection", frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()