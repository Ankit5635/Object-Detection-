import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import kagglehub

# Download dataset
dataset_path = kagglehub.dataset_download("alkanerturan/vehicledetection")

# Modify the dataset path based on actual folder structure
image_folder = os.path.join(dataset_path, "Test", "images")  # Change "Test" if needed

# Check if dataset exists
if not os.path.exists(image_folder):
    raise FileNotFoundError(f"Dataset not found at {image_folder}. Please check the path.")

print("Dataset path:", image_folder)

# Load TensorFlow model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
detector = model.signatures["serving_default"]

# Get image paths
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

if not image_paths:
    raise FileNotFoundError("No images found in the dataset folder. Please check the dataset.")

for image_path in image_paths:
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    input_tensor = tf.convert_to_tensor([image_rgb], dtype=tf.uint8)
    detections = detector(input_tensor)

    detection_scores = detections['detection_scores'][0].numpy()
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)

    height, width, _ = image.shape
    for i in range(len(detection_scores)):
        if detection_scores[i] > 0.5:
            y_min, x_min, y_max, x_max = detection_boxes[i]
            (startX, startY, endX, endY) = (int(x_min * width), int(y_min * height),
                                            int(x_max * width), int(y_max * height))
            
            cv2.rectangle(image_rgb, (startX, startY), (endX, endY), (0, 255, 0), 2)
            label = f"Object {detection_classes[i]}: {detection_scores[i]:.2f}"
            cv2.putText(image_rgb, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)

cv2.destroyAllWindows()
