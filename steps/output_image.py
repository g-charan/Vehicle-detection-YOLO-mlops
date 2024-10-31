from models.output import output_image
from zenml import step
import numpy

@step
def output_model(output_path:str,image: numpy.ndarray):
    op = output_image()
    op.output(output_path=output_path,image=image)