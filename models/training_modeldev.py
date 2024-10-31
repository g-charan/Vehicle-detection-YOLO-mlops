from ultralytics import YOLO
import numpy

from torch.utils.data import DataLoader




class yolo_detector:
    def train(self, data_path: str) -> YOLO:
        model = YOLO("yolov8n.pt")
        try:
            model.train(data=data_path, epochs=50,imgsz=640)
        except Exception as e:
            print(f"An error occurred during training: {e}")
        return model
    def predict(self,model: YOLO,image: numpy.ndarray) -> list:
        results = model.predict(image)
        return results