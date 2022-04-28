import copy
import time
from abc import abstractmethod
from queue import Empty

import cv2
import numpy as np
import torch
from PIL import ImageFont, Image, ImageDraw
from PyQt5.QtGui import QPixmap, QImage

from model.core.base_module import BaseModule, TASK_DATA_OK, DictData
from utils.vis import draw_keypoints136

box_color = (0, 255, 0)  #
cheating_box_color = (0, 0, 255)
draw_keypoints_default = False


# 绘制图片
def draw_frame(data, draw_keypoints=draw_keypoints_default, fps=-1):
    frame = data.frame.copy()  # 图片赋值给frame
    pred = data.detections  # 将data中处理后的数据赋值给pred
    preds_kps = data.keypoints  # 将data中的关键点赋值给preds_kps
    preds_scores = data.keypoints_scores  # 将data中的关键点成绩赋值给preds_scores
    if pred.shape[0] > 0:
        # 绘制骨骼关键点
        if draw_keypoints and preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        # 绘制目标检测框和动作分类
        frame_pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(frame_pil)
        for det, class_prob, best_pred in zip(pred, data.classes_probs, data.best_preds):
            det = det.to(torch.int)  # 转换类型
            class_name = data.classes_names[best_pred]  # 选择类名
            show_text = f"{class_name}"  # 设置要展示的类名
            show_color = box_color if best_pred == 0 else cheating_box_color  # 设置要展示颜色
            draw.rectangle((det[0], det[1], det[2], det[3]), outline=show_color, width=2)  # 绘制矩形
            # 文字
            fontText = ImageFont.truetype("resource/font/NotoSansCJKkr-Black.otf",
                                          int(40 * (min(det[2] - det[0], det[3] - det[1])) / 200),
                                          encoding="utf-8")  # 加载文字
            draw.text((det[0], det[1]), show_text, show_color, font=fontText)  # 在给定位置绘制字符串
        frame = np.asarray(frame_pil)
        # 头部姿态估计轴
        for (r, t) in data.head_pose:
            data.draw_axis(frame, r, t)
    # 绘制fps
    cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    data.frame_anno = frame  # 保存绘制过的图像


# 绘制模块
class DrawModule(BaseModule):
    def __init__(self):
        super(DrawModule, self).__init__()
        self.last_time = time.time()  # 最终时间

    def process_data(self, data):
        # 数据预处理
        print("显示图片")
        current_time = time.time()  # 当前时间
        fps = 1 / (current_time - self.last_time)  # fps计算
        frame = data.frame  # 框
        pred = data.detections
        preds_kps = data.keypoints
        preds_scores = data.keypoints_scores
        for det in pred:
            show_text = "person: %.2f" % det[4]
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), box_color, 2)  # 绘制矩形
            cv2.putText(frame, show_text,  # 绘制文字
                        (det[0], det[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        float((det[2] - det[0]) / 200),
                        box_color)
        if preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        # 记录fps
        cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        # 显示图像
        cv2.imshow("yolov5", frame)
        cv2.waitKey(40)
        self.last_time = current_time
        return TASK_DATA_OK

    def open(self):
        super(DrawModule, self).open()
        pass


# 保存模块
class FrameDataSaveModule(BaseModule):
    def __init__(self, app):
        super(FrameDataSaveModule, self).__init__()
        self.last_time = time.time()
        self.app = app

    def process_data(self, data):
        current_time = time.time()
        fps = 1 / (current_time - self.last_time)
        frame = data.frame
        pred = data.detections
        preds_kps = data.keypoints
        preds_scores = data.keypoints_scores
        for det in pred:
            show_text = "person: %.2f" % det[4]
            cv2.rectangle(frame, (det[0], det[1]), (det[2], det[3]), box_color, 2)
            cv2.putText(frame, show_text,
                        (det[0], det[1]),
                        cv2.FONT_HERSHEY_COMPLEX,
                        float((det[2] - det[0]) / 200),
                        box_color)
        if preds_kps is not None:
            draw_keypoints136(frame, preds_kps, preds_scores)
        # 记录fps
        cv2.putText(frame, "FPS: %.2f" % fps, (0, 52), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
        self.app.video_screen.setPixmap(self.cvImg2qtPixmap(frame))
        time.sleep(0.04)
        self.last_time = current_time
        return TASK_DATA_OK

    @staticmethod
    def cvImg2qtPixmap(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_height, image_width, image_depth = frame.shape
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        return QPixmap.fromImage(frame)

    def open(self):
        super(FrameDataSaveModule, self).open()
        pass


# 数据传输模块
class DataDealerModule(BaseModule):
    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(DataDealerModule, self).__init__(skippable=skippable)
        self.last_time = time.time()
        self.push_frame_func = push_frame_func
        self.last_data = None
        self.interval = interval  # 间隔
        self.size_waiting = True
        self.queue_threshold = 10  # 队列的阈值

    # 跳过大量数据
    @abstractmethod
    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        pass

    # 绘制框
    @abstractmethod
    def draw_frame(self, data, fps):
        pass

    # 数据预处理
    def process_data(self, data):
        # hasattr() 函数用于判断对象是否包含对应的属性。
        if hasattr(data, 'skipped') and self.last_data is not None:
            data = self.deal_skipped_data(data, copy.copy(self.last_data))
        else:
            self.last_data = data
        current_time = time.time()
        interval = (current_time - self.last_time)
        fps = 1 / interval
        data.fps = fps
        self.draw_frame(data, fps=fps)
        data.interval = interval
        self.last_time = current_time  # 更新时间
        self.push_frame_func(data)
        if hasattr(data, 'source_fps'):
            time.sleep(1 / data.source_fps * (1 + self.self_balance_factor()))
        else:
            time.sleep(self.interval)
        return TASK_DATA_OK

    # 平衡因子
    def self_balance_factor(self):
        factor = max(-0.999, (self.queue.qsize() / 20 - 0.5) / -0.5)
        # print(factor)
        return factor

    # 产生工作数据
    def product_task_data(self):
        # print(self.queue.qsize(), self.size_waiting)
        if self.queue.qsize() == 0:
            self.size_waiting = True
        if self.queue.qsize() > self.queue_threshold or not self.size_waiting:
            self.size_waiting = False
            try:
                task_data = self.queue.get(block=True, timeout=1)
                return task_data
            except Empty:
                return self.ignore_task_data
        else:
            time.sleep(1)
            return self.ignore_task_data

    # 存入工作数据
    def put_task_data(self, task_data):
        self.queue.put(task_data)

    # 打开
    def open(self):
        super(DataDealerModule, self).open()
        pass


# 作弊检测模块
class CheatingDetectionVisModule(DataDealerModule):

    def __init__(self, push_frame_func, interval=0.06, skippable=False):
        super(CheatingDetectionVisModule, self).__init__(push_frame_func, interval, skippable)

    # 跳过大量数据
    def deal_skipped_data(self, data: DictData, last_data: DictData) -> DictData:
        frame = data.frame  # 帧图片
        data = last_data  # 最终数据
        data.skipped = None  # 跳过
        data.frame = frame  # 帧图片
        # 返回一个和源张量同shape、dtype和device的张量，与源张量不共享数据内存，但提供梯度的回溯
        data.detections = data.detections.clone()
        # 添加抖动 返回一个张量，其大小与填充区间上均匀分布的随机数的张量相同
        data.detections[:, :4] += torch.rand_like(data.detections[:, :4]) * 3
        return data

    # 绘制框
    def draw_frame(self, data, fps):
        draw_frame(data, fps=fps)
