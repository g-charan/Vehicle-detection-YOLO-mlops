from zenml import step
from models.loading_image import cv2_images
import numpy

@step
def load_image(image_path:str) -> numpy.ndarray:
    img = cv2_images()
    image = img.loading_images(image_path=image_path)
    return image
    