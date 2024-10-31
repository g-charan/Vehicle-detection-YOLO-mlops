import cv2
import numpy

class output_image:
    def output(self,output_path: str,image: numpy.ndarray):
        cv2.imwrite(output_path, image)