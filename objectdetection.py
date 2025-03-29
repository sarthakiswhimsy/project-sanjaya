import cv2  # Import OpenCV for image processing
import numpy as np  # Import NumPy for numerical operations

# Load YOLO pre-trained model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Load weights and configuration

# Load the COCO dataset labels (80 classes)
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")  # Read class names from file

# Get the output layer names from YOLO
layer_names = net.getLayerNames()
out_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 for default camera

while True:
    ret, frame = cap.read()  # Capture frame from webcam
    if not ret:
        break  # Exit loop if the frame is not captured
    
    height, width, channels = frame.shape  # Get frame dimensions
    
    # Convert frame to a blob (preprocessing for YOLO model)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)  # Set blob as input to the network
    outputs = net.forward(out_layers)  # Perform forward pass to get predictions
    
    boxes = []  # List to store bounding boxes
    confidences = []  # List to store confidence scores
    class_ids = []  # List to store class IDs
    
    for output in outputs:
        for detection in output:
            scores = detection[5:]  # Get class probabilities
            class_id = np.argmax(scores)  # Get class with highest probability
            confidence = scores[class_id]  # Get confidence score of detected object
            
            if confidence > 0.3:  # Lowered threshold to capture more objects
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                x = int(center_x - w / 2)  # Compute top-left x coordinate
                y = int(center_y - h / 2)  # Compute top-left y coordinate
                
                boxes.append([x, y, w, h])  # Store bounding box
                confidences.append(float(confidence))  # Store confidence score
                class_ids.append(class_id)  # Store class ID
    
    # Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"  # Create label with confidence
            
            # Draw bounding box and label on frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow("YOLO Object Detection", frame)  # Display the output frame
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
   
