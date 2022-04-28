import threading

import numpy as np
import torch

from estimator.action_analysis import CheatingActionAnalysis
from estimator.classroom_action_classifier import ClassroomActionClassifier
from estimator.pose_estimator import PnPPoseEstimator
from model.core.base_module import TASK_DATA_OK, BaseModule

peep_threshold = -60  # 低头阈值


class CheatingActionModule(BaseModule):
    raw_class_names = ["seat", "write", "stretch", "hand_up_R", "hand_up_L",
                       "hand_up_highly_R", "hand_up_highly_L",
                       "relax", "hand_up", "pass_R", "pass_L", "pass2_R", "pass2_L",
                       "turn_round_R", "turn_round_L", "turn_head_R", "turn_head_L",
                       "sleep", "lower_head"]  # 未经处理的类名
    class_names = ["正常", "传纸条", "低头偷看", "东张西望"]  # 类名
    use_keypoints = [x for x in range(11)] + [17, 18, 19]  # 使用的关键点
    class_of_passing = [9, 10, 11, 12]
    class_of_peep = [18]
    class_of_gazing_around = [13, 14, 15, 16]

    # 0 正常坐姿不动
    # 1 正常低头写字
    # 2 正常伸懒腰
    # 3 举右手低
    # 4 举左手低
    # 5 举右手高
    # 6 举左手高
    # 7 起立
    # 8 抬手
    # 9 右伸手
    # 10 左伸手
    # 11 右伸手2
    # 12 左伸手2
    # 13 右转身
    # 14 左转身
    # 15 右转头
    # 16 左转头
    # 17 上课睡觉
    # 18 严重低头

    def __init__(self, weights, device='cpu', img_size=(480, 640), skippable=True):
        super(CheatingActionModule, self).__init__(skippable=skippable)
        self.weights = weights  # 模型权重
        self.classifier = ClassroomActionClassifier(weights, device)  # 设置分类器
        self.pnp = PnPPoseEstimator(img_size=img_size)  # 头部姿势估计

    def process_data(self, data):
        print("classroom_action_module is running,Thread is "+str(threading.current_thread()))
        print("classroom_action_module using data is " + str(data))
        print("~~~~~~~~~~~~~~")
        # 处理数据
        data.num_of_cheating = 0  # 作弊数目
        data.num_of_normal = 0  # 正常数目
        data.num_of_passing = 0  # 传纸条数目
        data.num_of_peep = 0  # 偷看数目
        data.num_of_gazing_around = 0  # 左右偷看数目
        if data.detections.shape[0] > 0:
            # 行为识别
            data.classes_probs = self.classifier.classify(data.keypoints[:, self.use_keypoints])  # 获取判断后的数据
            # 最佳行为分类
            """
            (d0,d1,…,dn−1) ，那么dim=0就表示对应到d0 也就是第一个维度，dim=1表示对应到也就是第二个维度，依次类推。
            torch.argmax(dim)会返回dim维度上张量最大值的索引。
            """
            data.raw_best_preds = torch.argmax(data.classes_probs, dim=1)  # 返回指定维度最大值的序号
            data.best_preds = [self.reclassify(idx) for idx in data.raw_best_preds]  # 获取动作的标志
            data.raw_classes_names = self.raw_class_names   # 未经处理的类名
            data.classes_names = self.class_names  # 类名
            # 头背部姿态估计
            # 获取面部返回旋转和平移向量
            data.head_pose = [self.pnp.solve_pose(kp) for kp in data.keypoints[:, 26:94, :2].numpy()]
            data.draw_axis = self.pnp.draw_axis   # 画出坐标轴
            data.head_pose_euler = [self.pnp.get_euler(*vec) for vec in data.head_pose]  # 获取所有的欧拉角
            # 传递动作识别
            is_passing_list = CheatingActionAnalysis.is_passing(data.keypoints)
            # 头部姿态辅助判断转头
            for i in range(len(data.best_preds)):  # 循环判断所有动作
                if data.best_preds[i] == 0:    # 如果是其他行为
                    if is_passing_list[i] != 0:  # 如果有传递行为
                        data.best_preds[i] = 1  # 判为传纸条
                    elif data.head_pose_euler[i][1][0] < peep_threshold:  # 如果小于预定的阈值
                        data.best_preds[i] = 2  # 判为低头偷看
            data.pred_class_names = [self.class_names[i] for i in data.best_preds]  # 行为名称
            # 统计人数
            data.num_of_normal = data.best_preds.count(0)  # 正常人数
            data.num_of_passing = data.best_preds.count(1)  # 传递行为人数
            data.num_of_peep = data.best_preds.count(2)  # 低头人数
            data.num_of_gazing_around = data.best_preds.count(3)  # 左右偷看人数
            data.num_of_cheating = data.detections.shape[0] - data.num_of_normal  # 作弊人数

        return TASK_DATA_OK

    def reclassify(self, class_idx):
        # 重新检测
        if class_idx in self.class_of_passing:  # 如果有伸手行为
            return 1
        elif class_idx in self.class_of_peep:  # 如果低头
            return 2
        elif class_idx in self.class_of_gazing_around:  # 如果左右偷看
            return 3
        else:  # 其余正常
            return 0

    def open(self):
        super(CheatingActionModule, self).open()