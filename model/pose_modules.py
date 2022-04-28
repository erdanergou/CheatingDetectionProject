import threading

from estimator.pose_estimator import AlphaPoseEstimator
from model.core.base_module import BaseModule, TASK_DATA_OK


class AlphaPoseModule(BaseModule):
    # 姿态检测模型
    def __init__(self, weights, device, skippable=True,
                 face_aligner_weights=None):
        super(AlphaPoseModule, self).__init__(skippable=skippable)
        self.weights = weights  # 模型权重
        self.pose_estimator = AlphaPoseEstimator(weights, device, face_aligner_weights=face_aligner_weights)

    def process_data(self, data):
        # 数据预处理
        print("pose_modules is running,Thread is "+str(threading.current_thread()))
        print("pose_modules using data is " + str(data))
        print("~~~~~~~~~~~~~~")
        preds_kps, preds_scores = self.pose_estimator.estimate(data.frame, data.detections)  # 获取处理后的数据
        data.keypoints = preds_kps  # 设置关键点为处理后的值
        data.keypoints_scores = preds_scores  # 设置关键点为处理后的值
        return TASK_DATA_OK

    def open(self):
        super(AlphaPoseModule, self).open()


