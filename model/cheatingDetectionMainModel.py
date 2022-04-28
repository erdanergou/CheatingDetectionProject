import csv
import os
import time
from itertools import islice
from threading import Thread, Lock

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QMainWindow

from model.core.base_module import DictData

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置

from model.classroom_action_module import CheatingActionModule
from model.core.task_solution import TaskSolution
from model.pose_modules import AlphaPoseModule
from model.video_modules import VideoModule
from model.vis_modules import CheatingDetectionVisModule
from model.yolo_modules import YoloV5Module
from model.list_items import VideoSourceItem, FrameData
from model.realTimeCapture import RealTimeCatchItem
from ui.CheatingDetection import Ui_MainWindow
from utils.common import second2str, OffsetList

yolov5_weight = './weights/yolov5s.torchscript.pt'
alphapose_weight = './weights/halpe136_mobile.torchscript.pth'
classroom_action_weight = './weights/classroom_action_lr_front_v2_sm.torchscript.pth'
device = 'cuda'


class CheatingDetectionApp(QMainWindow, Ui_MainWindow):
    # 作弊检查
    add_cheating_list_signal = QtCore.pyqtSignal(DictData)  # 添加作弊列表信号
    push_frame_signal = QtCore.pyqtSignal(DictData)  # 放入帧信号

    def __init__(self, parent=None):
        super(CheatingDetectionApp, self).__init__(parent)
        self.setupUi(self)
        self.video_source = 0  # 视频资源数
        self.frame_data_list = OffsetList()  # 帧数据列表
        self.opened_source = None  # 打开资源
        self.playing = None  # 放映选项
        self.playing_real_time = False  # 实时播放时间
        self.num_of_passing = 0  # 传递次数
        self.num_of_peep = 0  # 偷看次数
        self.num_of_gazing_around = 0  # 东张西望次数
        # 视频事件
        # 设置视频源事件
        self.open_source_lock = Lock()  # 开启资源锁
        # 点击事件(当点击开启资源时）
        # 当视频源输入的文本为空时调用摄像头，否则播放视频
        self.open_source_btn.clicked.connect(
            lambda: self.open_source(self.video_source_txt.text() if len(self.video_source_txt.text()) != 0 else 0))
        # 视频源列表点击事件
        self.video_resource_list.itemClicked.connect(lambda item: self.open_source(item.src))

        # 视频源关闭事件
        self.close_source_btn.clicked.connect(self.close_source)
        # 播放视频事件
        self.play_video_btn.clicked.connect(self.play_video)
        # 停止播放事件
        self.stop_playing_btn.clicked.connect(self.stop_playing)
        # 视频播放时间更改设置
        self.video_process_bar.valueChanged.connect(self.change_frame)
        # 将信号连接到指定槽函数
        self.push_frame_signal.connect(self.push_frame)

        # 设置列表
        self.add_cheating_list_signal.connect(self.add_cheating_list)
        # 初始化视频源
        self.init_video_source()

    def init_cheating_img_data(self):
        # 初始化作弊数据
        self.cheating_list_time = []
        self.cheating_list_count_data = dict(
            传纸条=[],
            低头偷看=[],
            东张西望=[]
        )

    def init_video_source(self):
        # 添加视频通道
        VideoSourceItem(self.video_resource_list, "摄像头", 0).add_item()  # 在列表中添加条目
        # 添加本地视频文件
        local_source = 'resource/videos'  # 本地路径
        if not os.path.exists(local_source):
            # 如果没有该路径，创建该路径
            os.makedirs(local_source)
        else:
            print(f"本地视频目录已创建: {local_source}")
        with open('resource/video_sources.csv', 'r', encoding='utf-8') as f:
            # 尝试打开
            reader = csv.reader(f)  # 读取文件
            for row in islice(reader, 1, None):  # islice()获取迭代器的切片，消耗迭代器
                # 添加条目
                VideoSourceItem(self.video_resource_list, row[0], row[1],).add_item()

    def open_source(self, source):
        # 打开资源
        self.open_source_lock.acquire(blocking=True)  # 设置锁为阻塞
        if self.opened_source is not None:  # 如果没有资源
            self.close_source()  # 关闭资源
        # 加载
        frame = np.zeros((480, 640, 3), np.uint8)  # 返回来一个给定形状和类型的用0填充的数组
        (f_w, f_h), _ = cv2.getTextSize("Loading", cv2.FONT_HERSHEY_TRIPLEX, 1, 2)  # 计算文本字符串的宽度和高度
        cv2.putText(frame, "Loading", (int((640 - f_w) / 2), int((480 - f_h) / 2)),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    1, (255, 255, 255), 2)  # 设置文本
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图片从bgr改为rgb
        frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
        image_height, image_width, image_depth = frame.shape  # 帧图片形状
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.video_screen.setPixmap(QPixmap.fromImage(frame))  # 展示图片

        # 启动视频源
        def open_source_func(self):
            fps = 12
            # 设置模型（资源模型，
            self.opened_source = TaskSolution() \
                .set_source_module(VideoModule(source, fps=fps)) \
                .set_next_module(YoloV5Module(yolov5_weight, device)) \
                .set_next_module(AlphaPoseModule(alphapose_weight, device)) \
                .set_next_module(CheatingActionModule(classroom_action_weight)) \
                .set_next_module(CheatingDetectionVisModule(lambda d: self.push_frame_signal.emit(d)))
            self.opened_source.start()
            self.playing_real_time = True
            self.open_source_lock.release()

        Thread(target=open_source_func, args=[self]).start()

    def close_source(self):
        # 关闭资源
        if self.opened_source is not None:  # 如果资源已经开启
            self.stop_playing()  # 停止播放
            self.opened_source.close()  # 关闭资源
            self.opened_source = None  # 设置打开的资源为none
            self.frame_data_list.clear()  # 清空框架数据列表
            self.video_process_bar.setMaximum(-1)  # 设置滑动条控件的最大值为-1
            self.playing_real_time = False  # 设置实时播放为False
            self.cheating_list.clear()  # 清空列表框
            self.real_time_catch_list.clear()  # 清空列表框
            self.init_cheating_img_data()  # 初始化作弊数据

    def push_frame(self, data):
        # 存放视频帧
        try:
            max_index = self.frame_data_list.max_index()  # 获取帧列表的最大索引
            # 如果帧列表不为空，则获取时间，否则为0
            time_process = self.frame_data_list[max_index].time_process if len(self.frame_data_list) > 0 else 0
            data.time_process = time_process + data.interval  # 设置时间为获取到的时间+间隔
            # 添加帧到视频帧列表
            self.frame_data_list.append(data)
            while len(self.frame_data_list) > 500:  # 如果列表中的图片>500
                self.frame_data_list.pop()  # 将第一张图片放出
            self.video_process_bar.setMinimum(self.frame_data_list.min_index())  # 设置滑动块的最小值
            self.video_process_bar.setMaximum(self.frame_data_list.max_index())  # 设置滑动块的最大值

            # 添加到作弊列表
            data.frame_num = max_index + 1  # 帧数目+1
            if data.num_of_cheating > 0 and self.check_cheating_change(data):  # 如果作弊人数>0并且作弊人数有变化
                self.add_cheating_list_signal.emit(data)  # 发送信号给调用add_cheating_list方法以添加到作弊列表

            # 判断是否进入实时播放状态
            if self.playing_real_time:
                self.video_process_bar.setValue(self.video_process_bar.maximum())

        except Exception as e:
            print(e)

    def check_cheating_change(self, data):
        # 检查作弊变化
        cond = all([self.num_of_passing >= data.num_of_passing,
                    self.num_of_peep >= data.num_of_peep,
                    self.num_of_gazing_around >= data.num_of_gazing_around])
        self.num_of_passing = data.num_of_passing
        self.num_of_peep = data.num_of_peep
        self.num_of_gazing_around = data.num_of_gazing_around

        return not cond

    def playing_video(self):
        # 播放视频
        try:
            while self.playing is not None and not self.playing_real_time:  # 如果playing不是none并且实时播放
                current_frame = self.video_process_bar.value()  # 获取滑动条控件的值
                max_frame = self.video_process_bar.maximum()  # 滑动条控件的最大值
                if current_frame < 0:  # 如果当前值小于0
                    continue  # 暂停
                elif current_frame < max_frame:  # 如果当前值小于最大值即没有播放完毕
                    data = self.frame_data_list[current_frame]  # 数据值为框架数据列表中当前值
                    if current_frame < max_frame:  # 如果当前值小于最大值即没有播放完毕
                        self.video_process_bar.setValue(current_frame + 1)  # 将滑动条件控件的值+1
                    time.sleep(data.interval)  # 睡眠时间为之前设置的时间间隔
                else:  # 播放完毕
                    self.stop_playing()  # 停止播放
                    self.playing_real_time = True  # 将实时播放设置为True
        except Exception as e:
            print(e)

    def stop_playing(self):
        # 停止播放
        if self.playing is not None:
            self.playing = None  # 将当前播放设置为none

    def add_cheating_list(self, data):
        try:
            # 添加作弊发生列表
            FrameData(self.cheating_list, data).add_item()
            # 作弊列表数量限制
            while self.cheating_list.count() > self.cheating_list_spin.value():
                self.cheating_list.takeItem(0)
            # 添加实时抓拍
            frame = data.frame  # 设置图片
            detections = data.detections  # 设置检测的结果
            cheating_types = data.pred_class_names  # 作弊类型
            time_process = data.time_process  # 时间
            frame_num = data.frame_num  # 图片数目
            best_preds = data.best_preds  # 最佳行为
            for detection, cheating_type, best_pred in zip(detections, cheating_types, best_preds):
                if best_pred == 0:
                    continue
                detection = detection[:4].clone()  # 检测边框
                detection[2:] = detection[2:] - detection[:2]
                RealTimeCatchItem(self.real_time_catch_list, frame, detection, time_process, cheating_type,
                                  frame_num).add_item()  # 添加
            # 实时抓拍列表限制
            real_time_catch_list_count = self.real_time_catch_list.count()  # 抓拍数目
            while real_time_catch_list_count > self.real_time_catch_spin.value():  # 如果抓拍数目超出限制
                self.real_time_catch_list.takeItem(real_time_catch_list_count - 1)  # 取一个项返回并从部件中删除该项
                real_time_catch_list_count -= 1  # 总数-1
        except Exception as e:
            print(e)

    def play_video(self):
        # 播放视频
        if self.playing is not None:  # 如果播放不为none
            return
        self.playing = Thread(target=self.playing_video, args=())
        self.playing.start()

    def change_frame(self):
        # 改变帧
        try:
            if len(self.frame_data_list) == 0:  # 如果帧数据列表为0
                return
            current_frame = self.video_process_bar.value()  # 当前帧为当前滑动块的值
            max_frame = self.video_process_bar.maximum()  # 最大帧数为滑动块最大值
            self.playing_real_time = current_frame == max_frame  # 是否开启实时播放即如果当前帧为最大帧
            # 更新界面
            data = self.frame_data_list[current_frame]  # 值为当前帧值
            maxData = self.frame_data_list[max_frame]  # 最大值为最大帧
            frame = data.frame_anno if self.show_box_ckb.isChecked() else data.frame  # 如果选择显示边框
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将bgr改为rgb
            frame = cv2.resize(frame, (self.video_screen.width() - 9, self.video_screen.height() - 9))  # 调整图像大小
            image_height, image_width, image_depth = frame.shape  # 获取边框的样式
            frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                           image_width * image_depth,
                           QImage.Format_RGB888)
            self.video_screen.setPixmap(QPixmap.fromImage(frame))  # 用于显示图像
            # 显示时间
            current_time_process = second2str(data.time_process)  # 当前时间
            max_time_process = second2str(maxData.time_process)  # 最大时间

            self.time_process_label.setText(f"{current_time_process}/{max_time_process}")
        except Exception as e:
            print(e)

    def close(self):
        # 关闭
        self.close_source()  # 关闭

    def open(self):
        # 打开
        pass
