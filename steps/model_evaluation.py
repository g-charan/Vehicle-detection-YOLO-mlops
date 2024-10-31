from zenml import step
from models.evaluation import evaluate_model
import numpy
from typing import Tuple
from typing_extensions import Annotated
import mlflow

from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker



@step(experiment_tracker=experiment_tracker.name)
def evaluating_model(results: list,image: numpy.ndarray) -> Tuple[Annotated[str,"traffic_pred"],Annotated[numpy.ndarray,"image"]]:
    eval = evaluate_model()
    vehicle_count,image = eval.eval(results,image)
    mlflow.log_metric("vehicle_count", vehicle_count)
    traffic_pred = eval.signal_pred(vehicle_count)
    image_rgb = eval.img_convert(image)
    return traffic_pred,image