import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO('best.pt')

def detect_objects(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Perform detection
    results = model(img)

    # Iterate through the detected results
    for result in results:
        for box in result.boxes:
            # Extract the coordinates and other details
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            confidence = box.conf[0].item()         # Convert to Python float
            class_id = int(box.cls[0].item())       # Convert to Python int

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{model.names[class_id]}: {confidence:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the output image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Example usage
detect_objects('pic3.jpg')
