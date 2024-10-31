from zenml import pipeline
from steps.training_model import trained_model
from steps.load_model import loading_model
from steps.load_image import load_image
from steps.predictions import predict_yolo
from steps.model_evaluation import evaluating_model
from steps.output_image import output_model

@pipeline(enable_cache=True)
def pipeline_dev(data_path:str,model_path: str,image_path: str,output_path: str):
    model = trained_model(data_path)
    loaded_model = loading_model(model_path)
    loaded_image = load_image(image_path)
    results = predict_yolo(loaded_model,loaded_image)
    predictions,modified_image = evaluating_model(results,loaded_image)
    output_model(output_path,modified_image)
    
    