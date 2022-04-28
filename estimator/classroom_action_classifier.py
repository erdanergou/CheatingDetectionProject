import torch


class ClassroomActionClassifier:
    # 课堂行为分类器
    def __init__(self, weights, device):
        self.device = device  # 选择设备
        self.model = torch.jit.load(weights).to(device)  # 加载模型

    # 将关键点数据处理成为一维数据返回
    @staticmethod
    def preprocess(keypoints):  # 预处理
        """
        torch.min(input， dim， keepdim=False， *， out=None)
        返回一个命名变量，其中是给定维度中张量的每一行的最小值。并且是找到的每个最小值的索引位置
        dim  要减小的尺寸。

        keypoints[:, :, 0] 第一列中所有的数，并成一行排列
        keypoints[:, :, 1] 第二列中所有的数，并成一行排列
        torch.min(keypoints[:, :, 0], dim=1).values 返回最小的值


        torch.stack() dim:要插入的尺寸。必须介于 0 和串联张量的维数之间（包括 0 和维数）
        沿新维度连接一系列张量。所有张量都需要具有相同的大小。

        torch.unsqueeze()函数用来在张量的某个位置上增加一个长度为1的维度 dim为起始的维度

        flatten 展平一个连续范围的维度，输出类型为Tensor start_dim (int) – 展平的开始维度
        :param keypoints:
        :return:
        """
        x_min = torch.min(keypoints[:, :, 0], dim=1).values  # 第一列最小值
        y_min = torch.min(keypoints[:, :, 1], dim=1).values  # 第二列最小值
        x_max = torch.max(keypoints[:, :, 0], dim=1).values  # 第一列最大值
        y_max = torch.max(keypoints[:, :, 1], dim=1).values  # 第二列最大值
        x1_y1 = torch.stack([x_min, y_min], dim=1).unsqueeze(1)  # 将第一列，第二列最小值合并在一起
        width = torch.stack([x_max - x_min, y_max - y_min], dim=1).unsqueeze(1)  # 计算长度、宽度
        scaled_keypoints = (keypoints - x1_y1) / width  # 尺寸数据
        scaled_keypoints = (scaled_keypoints - 0.5) / 0.5  # 计算数据
        return scaled_keypoints.flatten(start_dim=1)   # 将数据展平后返回

    def classify(self, keypoints):
        """
        分类
        将数据处理后阻断反向传播并将数据移至cpu然后返回
        :param keypoints:
        :return:
        """
        return self.model(self.preprocess(keypoints)).detach().cpu()
