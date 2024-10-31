from ultralytics import YOLO
class load_yolo:
    def load_train(self,model_path: str) -> YOLO:
        model = YOLO(model_path)
        return model