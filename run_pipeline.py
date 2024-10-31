from zenml import pipeline, step
from pipelines.training_pipeline import pipeline_dev
from zenml.client import Client



if __name__ == "__main__":
    # data_path="traffic-project-2/data.yaml",
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    pipeline_dev(data_path="traffic-project-2/data.yaml",model_path="pt_model/yolov8n.pt",image_path="1.jpg",output_path="output_image.jpg")
    
    
