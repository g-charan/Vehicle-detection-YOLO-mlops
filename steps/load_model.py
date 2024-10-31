from zenml import step
from models.loading_model import load_yolo
from ultralytics import YOLO

@step
def loading_model(model_path:str) -> YOLO :
    lom = load_yolo()
    model = lom.load_train(model_path)
    return model