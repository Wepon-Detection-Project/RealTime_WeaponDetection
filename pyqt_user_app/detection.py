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
    changePixmap = pyqtSignal(QPixmap)  # Define the changePixmap signal with QPixmap as an argument

    def __init__(self, token, location, receiver):
        super(Detection, self).__init__()

        self.token = token
        self.location = location
        self.receiver = receiver
        self.running = True  # Initialize the running flag

    # Runs the detection model, evaluates detections, and draws boxes around detected objects
    def run(self):
        # Load YOLO model here with proper configuration and weights
        net = cv2.dnn.readNet(r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\cfg\yolov4.cfg",
                              r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\weights\yolov4.weights")

        classes = []
        font = cv2.FONT_HERSHEY_SIMPLEX
        starting_time = time.time()

        # Loads object names
        with open(r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        # Set camera capture or video file path
        cap = cv2.VideoCapture(0)  # Use 0 for the default camera, or provide the path for a video file
        if not cap.isOpened():
            print("Error: Failed to open the camera.")
            return

        while self.running:
            # Capture video frame
            ret, frame = cap.read()

            if not ret:
                # If there is an issue with capturing the frame, stop the detection
                print("Error: Failed to capture frame from the camera.")
                break

            # Preprocess the frame for YOLO model
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)

            # Run forward pass to get output layer names and predictions
            output_layers = net.getUnconnectedOutLayersNames()  # Get the names of the output layers
            outs = net.forward(output_layers)

            # Perform object detection and draw bounding boxes on the frame
            # Implement this part based on the YOLO model output

            # Convert the OpenCV image to QImage
            height, width, channels = frame.shape
            bytesPerLine = channels * width
            q_image = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(q_image)

            # Emit the changePixmap signal to update the GUI
            self.changePixmap.emit(pixmap)

            # Evaluating detections
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    # If detection confidence is above 98% a weapon was detected
                    if confidence > 0.98:
                        # Calculating coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

            # Draw boxes around detected objects
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (256, 0, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label + " {0:.1%}".format(confidence), (x, y - 20), font, 3, color, 3)

                    elapsed_time = starting_time - time.time()

                    # Save detected frame every 10 seconds
                    if elapsed_time <= -10:
                        starting_time = time.time()
                        self.save_detection(frame)

            # Showing final result
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bytesPerLine = channels * width
            convertToQtFormat = QImage(rgbImage.data, width, height, bytesPerLine, QImage.Format_RGB888)
            p = convertToQtFormat.scaled(854, 480, Qt.KeepAspectRatio)

            # Convert QImage to QPixmap
            pixmap = QPixmap.fromImage(p)

            # Emit the changePixmap signal to update the GUI
            self.changePixmap.emit(pixmap)

        cap.release()

    # Saves detected frame as a .jpg within the saved_alert folder
    def save_detection(self, frame):
        cv2.imwrite(r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\saved_frames\frame.jpg", frame)
        print('Frame Saved')
        self.post_detection()

    # Sends an alert to the server
    def post_detection(self):
        url = 'http://127.0.0.1:8000/api/images/'
        headers = {'Authorization': 'Token ' + self.token}
        files = {'image': open(r"C:\Users\act\OneDrive\Desktop\Harshit\RealTime_weapon_Detection\pyqt_user_app\saved_frames\frame.jpg", 'rb')}
        data = {'user_ID': self.token, 'location': self.location, 'alert_receiver': self.receiver}
        response = requests.post(url, files=files, headers=headers, data=data)

        # HTTP 200
        if response.ok:
            print('Alert was sent to the server')
        # Bad response
        else:
            print('Unable to send alert to the server')
