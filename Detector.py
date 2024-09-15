import cv2

class Detector:
    def __init__(self, videoCapture, configPath, modelPath, classesPath):
        self.videoCapture = videoCapture
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # Load the model
        self.net = cv2.dnn_DetectionModel(self.modelPath, self.configPath)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        # Read the class names
        self.readClasses()

    def readClasses(self):
        # Read the class names from the file
        with open(self.classesPath, 'r') as f:
            self.classesList = f.read().splitlines()
            print(f"{len(self.classesList)} classes loaded.")

    def detect(self, frame):
        # Perform object detection on the given frame
        classIds, confs, bbox = self.net.detect(frame, confThreshold=0.5)

        # Process the detected objects
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                # Get the name of the detected class
                label = self.classesList[classId - 1]

                # Draw a rectangle and add the label with confidence score
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f"{label} {confidence * 100:.2f}%",
                            (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return frame  # Return the processed frame as output
