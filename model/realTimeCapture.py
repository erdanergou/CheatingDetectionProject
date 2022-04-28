import cv2
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QListWidget
from ui.realTimeCapture import Ui_RealTimeCatch
from utils.common import second2str
from utils.img_cropper import CropImage
import os


class RealTimeCatchItem(QListWidgetItem):
    # 实时捕获项目
    def __init__(self, list_widget: QListWidget, img, detection, time_process, cheating_type, frame_num):
        super(RealTimeCatchItem, self).__init__()
        self.list_widget = list_widget  # 列表部件
        self.widget = RealTimeCatchItem.Widget(list_widget)  # 小部件
        self.setSizeHint(QSize(200, 200))  # 获取推荐大小

        self.img = img  # 抓拍图片
        self.time_process = time_process  # 时间处理
        self.cheating_type = cheating_type  # 作弊类型
        self.frame_num = frame_num  # 框数
        self.detection = detection  # 检测

    def add_item(self):
        # 添加条目
        size = self.sizeHint()  # 推荐大小
        self.list_widget.insertItem(0, self)  # 插入到QListWidget即list_widget中
        self.widget.setSizeIncrement(size.width(), size.height())  # 使用基本大小来计算适当的自旋框大小
        self.list_widget.setItemWidget(self, self.widget)  # 设置显示我们自定义的QWidget
        # 设置图像
        catch_img = self.widget.catch_img  # 抓拍图像
        frame = CropImage.crop(self.img, self.detection, 1, catch_img.width(), catch_img.height())  # 获取裁剪后的图片
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将bgr改为rgb
        image_height, image_width, image_depth = frame.shape  # 图片的高、宽、通道数
        frame = QImage(frame.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                       image_width * image_depth,
                       QImage.Format_RGB888)
        self.widget.catch_img.setPixmap(QPixmap.fromImage(frame))  # 用于显示图像
        filePath = os.getcwd().replace("\\", "/")
        filePath = filePath + "/cheatingResult"
        frame.save(filePath + str(self.cheating_type) + str(second2str(self.time_process)), "JPG", -1)
        # 设置时间
        self.widget.time_lbl.setText(f'{second2str(self.time_process)}')  # 设置文本时间
        self.widget.cheating_type_lbl.setText(self.cheating_type)  # 设置文本文字

    class Widget(QWidget, Ui_RealTimeCatch):
        # 小部件
        def __init__(self, parent=None):
            super(RealTimeCatchItem.Widget, self).__init__(parent)
            self.setupUi(self)
