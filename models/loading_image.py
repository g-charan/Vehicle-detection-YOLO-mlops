import cv2
import numpy

class cv2_images:
    def loading_images(self,image_path:str) -> numpy.ndarray:
        try:
            image = cv2.imread(image_path)
            return image
        except Exception as ex:
            raise ex
