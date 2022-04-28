import threading

from estimator.yolo_detector import YoloV5Detector
from model.core.base_module import BaseModule, TASK_DATA_OK


class YoloV5Module(BaseModule):
    # yolo模型
    def __init__(self, weights, device, skippable=True):
        super(YoloV5Module, self).__init__(skippable=skippable)
        self.weights = weights  # 设置权重
        self.detector = YoloV5Detector(self.weights, device)  # 检测器

    def process_data(self, data):
        # 数据预处理
        print("yolo_modules is running,Thread is "+str(threading.current_thread()))
        print("yolo_modules using data is " + str(data))
        print("~~~~~~~~~~~~~~")
        data.detections = self.detector.detect(data.frame)  # 经过模型处理后的数据
        return TASK_DATA_OK

    def open(self):
        super(YoloV5Module, self).open()
