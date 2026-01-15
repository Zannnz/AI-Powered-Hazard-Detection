from ultralytics import YOLO
import cv2

class YOLOv8Detector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the YOLOv8 model."""
        self.model = YOLO(model_path)

    def detect_and_draw(self, frame, conf=0.25):
        """
        Runs inference on a frame and returns the frame with 
        bounding boxes and labels drawn.
        """
        # Run inference
        results = self.model.predict(frame, conf=conf, verbose=False)
        
        # results[0].plot() returns the BGR image with boxes drawn by Ultralytics
        annotated_frame = results[0].plot()
        
        return annotated_frame
