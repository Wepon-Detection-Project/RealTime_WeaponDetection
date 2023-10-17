from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import time
import requests

# Handles the YOLOv4 detection algorithm, saves detected frames, and sends an alert to the server-side application
class Detection(QThread):
    changePixmap = pyqtSignal(QImage)  # Define the changePixmap signal with QImage as an argument

    def __init__(self, token, location, receiver):
        super(Detection, self).__init__()
        self.token = token
        self.location = location
        self.receiver = receiver
        self.running = True

    def run(self):
        # Load YOLO model here with proper configuration and weights
        net = cv2.dnn.readNet(r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\cfg\yolov4.cfg",
                              r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\weights\yolov4.weights")

        # Set camera capture or video file path
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture(r'C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\ak47.mp4')  # For video file

        classes = []

        # Loads object names
        with open(r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        while self.running:
            # Capture video frame
            ret, frame = cap.read()

            if not ret:
                # If there is an issue with capturing the frame, stop the detection
                break

            # Preprocess the frame for YOLO model
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)

            # Run forward pass to get output layer names and predictions
            output_layers = net.getUnconnectedOutLayersNames()
            outs = net.forward(output_layers)

            # ... (rest of the code for object detection and drawing bounding boxes)

            # Convert the OpenCV image to QImage
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            q_image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            # Emit the changePixmap signal to update the GUI with QImage
            self.changePixmap.emit(q_image)

        cap.release()

# Manages detection window, starts and stops detection thread
class DetectionWindow(QMainWindow):
    def __init__(self):
        super(DetectionWindow, self).__init__()
        loadUi(r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\UI\detection_window.ui", self)

        self.detection = None  # Initializing the Detection instance

        self.stop_detection_button.clicked.connect(self.close)

    def create_detection_instance(self, token, location, receiver):
        self.detection = Detection(token, location, receiver)

    # Assigns detection output to the label to display detection output
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label_detection.setPixmap(QPixmap.fromImage(image))

    def start_detection(self):
        if self.detection:
            self.detection.changePixmap.connect(self.setImage)
            self.detection.start()
            self.show()
        else:
            print("Detection instance not created. Call 'create_detection_instance' before starting detection.")

    def closeEvent(self, event):
        if self.detection:
            self.detection.running = False
            event.accept()
