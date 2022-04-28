import numpy as np
import torch

"""
MobileNetV1就是把VGG中的标准卷积层换成深度可分离卷积
可分离卷积主要有两种类型：空间可分离卷积和深度可分离卷积。
空间可分离就是将一个大的卷积核变成两个小的卷积核
eg:
 [[1 , 2 ,3 ],[0, 0, 0],[2, 4, 6]]  = [1, 0 ,2]的转置 * [ 1, 2, 3]
深度可分离卷积就是将普通卷积拆分成为一个深度卷积和一个逐点卷积。
"""

"""
mobilefacenet其实是mobilenetV2的改进版本
"""


class MobileNetSEFaceAligner:
    """
    np.asarray:将结构数据转化为ndarray。
    reshape:形状
    """
    #
    mean = np.asarray([0.485, 0.456, 0.406]).reshape(1, 1, 1, 3)  # 均值
    std = np.asarray([0.229, 0.224, 0.225]).reshape(1, 1, 1, 3)  # 标准差
    """
    使用Imagenet的均值和标准差是一种常见的做法。它们是根据数百万张图像计算得出的。
    如果要在自己的数据集上从头开始训练，则可以计算新的均值和标准差。
    否则，建议使用Imagenet预设模型及其平均值和标准差。
    """
    def __init__(self, weights, device):
        self.device = device  # 设备
        self.model = torch.jit.load(weights).to(device)  # 加载模型

    def preprocess(self, faces):
        # 预处理
        faces = (faces / 255 - self.mean) / self.std
        return torch.from_numpy(faces.transpose((0, 3, 1, 2)))

    def align(self, faces):
        # 对齐
        result = self.model(self.preprocess(faces).to(device=self.device, dtype=torch.float32)).detach().cpu()
        return result.view(-1, 68, 2)
