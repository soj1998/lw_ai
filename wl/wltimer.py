import time
import numpy as np


class Timer:
    """ 记录多次运行的时间 """

    def __init__(self):
        self.tik = None
        self.times = []
        self.start()

    def start(self):
        """ 启动计时器 """
        self.tik = time.time()

    def stop(self):
        """ 停止计时器并将时间记录在列表 """
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """ 返回平均耗时 """
        return sum(self.times)/len(self.times)

    def sum(self):
        """ 返回全部耗时 """
        return sum(self.times)

    def cumsum(self):
        """ 返回累计时间 cumsum返回给定轴上元素的累积和。"""
        return np.array(self.times).cumsum().tolist()