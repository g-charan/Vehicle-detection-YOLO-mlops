from zenml import step
from models.training_modeldev import yolo_detector
from ultralytics import YOLO
import numpy


@step
def predict_yolo(model: YOLO,image: numpy.ndarray) -> list:
    pred = yolo_detector()
    results = pred.predict(model=model,image=image)
    return results
    