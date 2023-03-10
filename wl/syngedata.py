# coding=utf-8

import numpy as np
""" 自动生成数据集 """


class YiYuanSj:
    """ 一元一次函数 """
    def __init__(self, w=1, b=1):
        self.y = []
        self.x = None
        self.w = w
        self.b = b
        self.generate()

    def generate(self):
        self.x = np.random.rand(1000)
        for i in range(len(self.x)-20):
            y = self.x[i]*self.w + self.b
            self.y.append(y)
        for i in range(20):
            y = self.x[len(self.x)-i-1]*self.w + np.random.rand(1)[0]
            self.y.append(y)
