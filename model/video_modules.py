import time
import threading
import cv2

from model.core.base_module import BaseModule, TASK_DATA_CLOSE, TASK_DATA_OK, TaskData, TASK_DATA_SKIP, \
    TASK_DATA_IGNORE


class VideoModule(BaseModule):
    # 视频模型
    def __init__(self, source=0, fps=25, skippable=False):
        super(VideoModule, self).__init__(skippable=skippable)
        self.task_stage = None  # 任务阶段
        self.source = source  # 资源
        self.cap = None  # 摄像头
        self.frame = None  # 帧图片
        self.ret = False  # 是否读取到图片
        self.skip_timer = 0  # 跳过计时器
        self.set_fps(fps)  # 设置fps
        self.loop = True  # 设置循环

    def process_data(self, data):
        print("video_modules is running,Thread is "+str(threading.current_thread()))
        print("video_modules using data is " + str(data))
        print("~~~~~~~~~~~~~~")
        # 数据预处理
        if not self.ret:  # 如果没有读取到图片
            if self.loop:  # 如果循环
                self.open()  # 开启摄像头
                return TASK_DATA_IGNORE  # 返回值（数据忽视）
            else:
                return TASK_DATA_CLOSE  # 返回值（数据关闭）
        data.source_fps = self.fps  # 设置fps
        data.frame = self.frame  # 设置帧图片
        """
        ret, frame = cap.read()返回值含义：
        参数ret 为True 或者False,代表有没有读取到图片
        第二个参数frame表示截取到一帧的图片
        """
        self.ret, self.frame = self.cap.read()
        result = TASK_DATA_OK  # 设置结果为ok
        if self.skip_timer != 0:  # 如果跳过计时器不为0
            result = TASK_DATA_SKIP  # 结果为数据跳过
            data.skipped = None  # 设置跳过为none
        skip_gap = int(self.fps * self.balancer.short_stab_interval)  # 跳过差距
        if self.skip_timer > skip_gap:  # 如果跳过计时器的值大于跳过差距
            self.skip_timer = 0  # 设置跳过计时器为0
        else:
            self.skip_timer += 1  # 跳过计时器+1
        time.sleep(self.interval)  # 睡眠一段时间
        return result

    def product_task_data(self):
        # 产生工作数据
        return TaskData(self.task_stage)

    def set_fps(self, fps):
        # 设置fps
        self.fps = fps
        self.interval = 1 / fps

    def open(self):
        # 开启
        super(VideoModule, self).open()  # 调用父类的开启方法
        if self.cap is not None:  # 如果摄像头不为空
            self.cap.release()  # 释放内存
        self.cap = cv2.VideoCapture(self.source)  # 调用摄像头或打开视频
        if self.cap.isOpened():  # 如果已经打开
            self.set_fps(self.cap.get(cv2.CAP_PROP_FPS))  # 设置fps
            """
            ret, frame = cap.read()返回值含义：
            参数ret 为True 或者False,代表有没有读取到图片
            第二个参数frame表示截取到一帧的图片
            """
            self.ret, self.frame = self.cap.read()
            print("视频源帧率: ", self.fps)

