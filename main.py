from DistanceDetector import DistanceDetector
import os
import cv2

def main():
    videoCapture = cv2.VideoCapture(0)  # Capture video from the default camera

    # Set file paths from the model_data directory
    configPath = os.path.join("model_data", "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    modelPath = os.path.join("model_data", "frozen_inference_graph.pb")
    classesPath = os.path.join("model_data", "coco.names")

    # Focal length and known width (in cm)
    focal_length = 800  # Example: 800
    known_width = 14.0  # Known width of the object (in cm)

    # Check if the file paths exist
    if not os.path.exists(configPath):
        print("Config file not found.")
        return
    if not os.path.exists(modelPath):
        print("Model file not found.")
        return
    if not os.path.exists(classesPath):
        print("Classes file not found.")
        return

    print("All files successfully found.")

    # Create the distance detector object
    detector = DistanceDetector(videoCapture, configPath, modelPath, classesPath, focal_length, known_width)
    print("Model successfully loaded and configured.")

    while True:
        ret, frame = videoCapture.read()  # Read a frame from the video capture
        if not ret:
            break

        # Perform object detection and calculate distances
        processed_frame, distances = detector.detect_and_calculate_distance(frame)

        # Print the detected objects and their distances to the console
        for label, distance in distances:
            print(f"{label}: {distance:.2f} cm")

        # Display the processed frame
        cv2.imshow("Object Detection and Distance Measurement", processed_frame)

        # Break the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    videoCapture.release()  # Release the video capture resource
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == '__main__':
    main()
