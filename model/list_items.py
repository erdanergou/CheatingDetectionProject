from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QListWidgetItem, QWidget, QListWidget
from ui.cheatingListItem import Ui_CheatingListItem
from utils.common import second2str


# 帧数据
class FrameData(QListWidgetItem):

    def __init__(self, list_widget: QListWidget, data, filter_idx=0):
        super(FrameData, self).__init__()
        self.list_widget = list_widget  # 窗口小部件
        self.data_ = data  # 数据
        self.frame_num = data.frame_num  # 帧数
        self.widget = FrameData.Widget(list_widget)  # 小部件
        self.time_process = second2str(data.time_process)  # 计算时间

        color1 = '#ff0000' if data.num_of_passing > 0 else '#1f1f1f'  # 定义颜色
        color2 = '#ff0000' if data.num_of_peep > 0 else '#1f1f1f'  # 定义颜色
        color3 = '#ff0000' if data.num_of_gazing_around > 0 else '#1f1f1f'  # 定义颜色
        # 设置展示的文字
        self.str = f"时间[{self.time_process}] " \
            f"<span style=\" color: {color1};\">传纸条: {data.num_of_passing}</span> " \
            f"<span style=\" color: {color2};\">低头偷看: {data.num_of_peep}</span> " \
            f"<span style=\" color: {color3};\">东张西望: {data.num_of_gazing_around}</span>"
        self.widget.lbl.setText(self.str)  # 将文字设置到lbl中
        idx = filter_idx
        if idx == 0:
            self.setHidden(True)  # 隐藏该控件
        elif idx == 1:
            self.setHidden(self.data_.num_of_passing == 0)  # 如果传递纸条的人数为零则隐藏
        elif idx == 2:
            self.setHidden(self.data_.num_of_peep == 0)  # 如果低头的人数为零则隐藏
        elif idx == 3:
            self.setHidden(self.data_.num_of_gazing_around == 0)  # 如果偷看的人数为零则隐藏
        self.setSizeHint(QSize(400, 46))  # 设置大小

    def add_item(self):
        # 添加条目
        size = self.sizeHint()  # 获取窗口推荐大小
        self.list_widget.addItem(self)  # 将该控件添加
        self.widget.setSizeIncrement(size.width(), size.height())  # 设置调整大小时的每次变化的增量大小
        self.list_widget.setItemWidget(self, self.widget)  # 设置显示我们自定义的QWidget

    class Widget(QWidget, Ui_CheatingListItem):
        # 窗口小部件
        def __init__(self, parent=None):
            super(FrameData.Widget, self).__init__(parent)
            self.setupUi(self)


class VideoSourceItem(QListWidgetItem):
    # 视频资源列表
    def __init__(self, list_widget, name, src):
        super(VideoSourceItem, self).__init__()
        self.setText(name)  # 设置文字
        self.src = src  # 来源
        self.list_widget = list_widget  # 列表小部件

    def add_item(self):
        # 添加条目
        self.list_widget.addItem(self)
