import cv2
import numpy as np
import torch
from torchvision import transforms

from utils.general import non_max_suppression


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # 在满足跨距多约束的情况下调整大小和填充图像
    shape = img.shape[:2]  # 目前的形状 [高, 宽]
    if isinstance(new_shape, int):  # 判断一个对象是否是一个已知的类型，会认为子类是一种父类类型，考虑继承关系。
        new_shape = (new_shape, new_shape)

    # 尺度比 (新 / 旧)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 只按比例缩小，不按比例放大(为了更好的测试mAP)
        r = min(r, 1.0)

    # 计算padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # round方法返回浮点数x的四舍五入值。
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding np.mod计算两数组对应位置元素的余数。
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度，高度 比

    dw /= 2  # 将填充分成两部分
    dh /= 2

    if shape[::-1] != new_unpad:  # 重新设置大小
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return img, ratio, (dw, dh)


class YoloV5Detector:
    # yolo检测器
    def __init__(self, weights, device):
        """
        初始化
        :param weights:  模型权重
        :param device:  运行的设备
        """
        self.device = device  # 运行设备
        self.model = torch.jit.load(weights).to(device)  # 加载模型
        self.conf_thres = 0.35  # conf界限
        self.iou_thres = 0.45   # iou算法界限
        self.agnostic_nms = False  # 怀疑的数目
        self.max_det = 1000  # 最大
        self.classes = [0]  # 级别
        self.transformer = transforms.Compose([transforms.ToTensor()])  # 串联多个图片变换的操作
        # 预热
        #_ = self.model(torch.zeros(1, 3, 640, 480).to(self.device))

    def preprocess_img(self, img):
        """
        预处理图片
        :param img:
        :return:
        img[：，：，：：-1]的作用就是实现RGB到BGR通道的转换 
        （若图片一开始就是BGR的，就是实现从BGR到RGB的转换）。
        unsqueeze()升维
        """
        return self.transformer(img[:, :, ::-1].copy()).unsqueeze(0).to(self.device, dtype=torch.float32)

    def detect(self, img):
        """
        检查
        :param img:图片
        :return:
        """
        # 预处理图片
        newimg = self.preprocess_img(img)
        # 使用模型处理图片
        pred = self.model(newimg)[0]
        """
         非极大值抑制，去掉多余的检测框
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        conf_thres:置信度即得分score的阈值，yolo为0.25。
        iou_thres：重叠度阈值，为0.45
        classes：类别数，可以设置保留哪一类的box
        agnostic_nms:是否去除不同类别之间的框,默认false
        max_det:一张图片中最大识别种类的个数，默认300

        """
        nms_pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                   max_det=self.max_det)
        """
        detach()是阻断反向传播的，且经过detach()方法后，变量仍然在GPU上
        cpu()将数据移至cpu中，则 可以对该Tensor数据进行一系列操作
        """
        newpred = nms_pred[0].detach().cpu()
        return newpred
