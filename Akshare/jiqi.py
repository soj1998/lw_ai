# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
from datetime import datetime
import numpy as np
from gm.api import *
import sys
from sklearn import svm

'''
本策略选取了七个特征变量组成了滑动窗口长度为15天的训练集，
随后训练了一个二分类（上涨/下跌）的支持向量机模型。
若没有仓位，则在每个星期一的时候输入标的股票近15个交易日的特征变量进行预测，
并在预测结果为上涨的时候购买标的股票。
若已经持有仓位，则在盈利大于10%的时候止盈，在星期五损失大于2%的时候止损。
特征变量为：1.收盘价/均值 2.现量/均值 3.最高价/均价 4.最低价/均价
5.现量 6.区间收益率 7.区间标准差
回测时间为：
2017-07-01 09:00:00 到2017-10-01 09:00:00
'''

def init(context):
    # 订阅浦发银行的分钟bar行情
    context.symbol = 'SHSE.600000'
    subscribe(symbols=context.symbol, frequency='60s')
    start_date = '2016-03-01' # SVM训练起始时间
    end_date = '2017-06-30' # SVM训练终止时间
    # 用于记录工作日
    # 获取目标股票的daily历史行情
    recent_data = history(context.symbol, frequency='id', start_time=start_date,
                          end_time=end_date, fill_missing='last', df=True)
    days_value = recent_data['bob'].values
    days_close = recent_data['close'].values
    days = []
    # 获取行情日期列表
    print('准备数据训练SVM')
    for i in range(len(days_value)):
        days.append(str(days_value[i])[0:10])
    x_all = []
    y_all = []
    for index in range(15, (len(days)-5)):
        start_day = days[index -15]
        end_day = days[index]
        data = history(context.symbol, frequency='id', start_time=start_day,
                       end_time=end_day, fill_missing='last',
                       df=True)
        close = data['close'].values
        max_x = data['high'].values
        min_n = data['low'].values
        amount = data['amount'].values
        volume = []
        for i in range(len(close)):
            volume_temp = amount[i]/close[i]
            volume.append(volume_temp)
        close_mean = close[-1]/np.mean(close)  # 收盘价/均值
        volume_mean = volume[-1]/np.mean(volume)  # 现量/均量
        max_mean = max_x[-1]/np.mean(max_x)  # 最高价/均价
        min_mean = min_n[-1]/np.mean(min_n)  # 最低价/均价
        vol = volume[-1]  # 现量
        return_now = close[-1]/close[0]  # 区间收益率
        std = np.std(np.array(close), axis=0)  # 区间标准差
        # 将计算出的指标添加到训练集X
        # features用于存放因子
        features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
        x_all.append(features)
    for i in range(len(days_value)):
        days.append(str(days_value[i])[0:10])

