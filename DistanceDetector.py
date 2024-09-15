import cv2
import numpy as np

class DistanceDetector:
    def __init__(self, videoCapture, configPath, modelPath, classesPath, focal_length, known_width):
        self.videoCapture = videoCapture
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath
        self.focal_length = focal_length  # Focal length of the camera
        self.known_width = known_width  # Known width of the object in real-world measurements

        # Load the model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(640, 640)  # Increase resolution if needed
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Read the class names
        self.readClasses()

    def readClasses(self):
        # Read the class names from the provided file
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
            print(f"{len(self.classesList)} classes loaded.")

    def detect_and_calculate_distance(self, frame):
        # Perform object detection on the frame
        classIds, confs, bbox = self.net.detect(frame, confThreshold=0.5)
        distances = []

        # Process detected objects
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                # Get the name of the detected class
                label = self.classesList[classId - 1]

                # Get the width of the detected object's bounding box (w)
                w = box[2]

                # Calculate distance in cm
                distance = self.calculate_distance(w)
                distances.append((label, distance))

                # Draw a rectangle around the detected object and display the distance
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"{label} {distance:.2f} cm",
                            (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame, distances

    def calculate_distance(self, width_in_pixels):
        # Calculate distance based on the width of the detected object in pixels
        if width_in_pixels > 0:
            distance = (self.known_width * self.focal_length) / width_in_pixels
            return distance
        else:
            return None
