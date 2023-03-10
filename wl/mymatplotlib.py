import matplotlib.pyplot as plt


class MyMatplotlib:
    """ 画图软件 """
    def __init__(self):
        self.pltins = None
        pass

    """ 画直线图 """
    def linearplot(self, x, y, linewidth=6):
        plt.plot(x, y, linewidth)
        self.pltins = plt.show()

    """ 画点状图 """
    def scatterplot(self, x, y):
        plt.scatter(x, y)
        self.pltins = plt.show()

    def closplot(self):
        plt.close(self.pltins)
