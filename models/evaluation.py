import cv2
import numpy
from typing import Tuple
from typing_extensions import Annotated
class evaluate_model:
    def eval(self,results:list,image: numpy.ndarray) -> Tuple[Annotated[int,"vehicle_count"],Annotated[numpy.ndarray,"image"]]:
        try:
            vehicle_count = 0
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  
                label = box.cls[0]  
                confidence = box.conf[0]  
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Blue box
                cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                vehicle_count += 1
            return vehicle_count,image
        except Exception as ex:
            raise ex
    def signal_pred(self,vehicle_count: int) -> str:
        try:
            if vehicle_count < 5:
                traffic_light = "Red"
            else:
                traffic_light = "Green"
            print(traffic_light)
            return traffic_light
        except Exception as ex:
            raise ex
    def img_convert(self,image: numpy.ndarray) -> numpy.ndarray:
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image_rgb
        except Exception as ex:
            raise ex
