import json
import re

import cv2
import numpy as np


def second2str(seconds):
    # 计算时间
    #  divmod() 函数把除数和余数运算结果结合起来，返回一个包含商和余数的元组(a // b, a % b)。
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


# 偏差列表
class OffsetList(list):
    def __init__(self, seq=()):
        super(OffsetList, self).__init__(seq)
        self.offset = 0

    def min_index(self):
        # 最小索引
        return self.offset

    def max_index(self):
        # 最大索引
        return self.offset + len(self) - 1

    def __getitem__(self, item):
        # 获取条目
        return super(OffsetList, self).__getitem__(max(0, item - self.offset))

    def append(self, __object) -> None:
        super(OffsetList, self).append(__object)

    def pop(self):
        self.offset += 1
        super(OffsetList, self).pop(0)

    def clear(self) -> None:
        self.offset = 0
        super(OffsetList, self).clear()


# 正则表达式
zhPattern = re.compile(u'[\u4e00-\u9fa5|\d|a-z|A-Z]+')


# 判断文件名的正确性
def validFileName(filename):
    return True if zhPattern.fullmatch(filename) else False


# 读取面部图片
def read_mask_img(filename):
    mask = cv2.imread(filename)
    mask[mask > 128] = 255
    mask[mask <= 128] = 0
    mask = [[True if c[0] == 255 else False for c in r] for r in mask]
    return mask


# 读取json文件
def read_encoding_json2npy(path):
    with open(path) as f:
        return np.array(json.load(f))


