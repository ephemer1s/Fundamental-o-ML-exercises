# -*- coding: utf-8 -*-


import numpy as np
from scipy import signal
import math
import matplotlib.pylab as plt
import os
import time
import argparse


def arg_parse():
    '''
    获取运行参数，-f为文件序号，113或者114，-r为数据行数，可选范围为1，2，3，4
    返回一个可被调用的参数对象
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename","-f", dest='filename'
                        , help="要读取的文件序号", default="113", type=str)
    parser.add_argument("--row","-r", dest='row'
                        , help="选择第几行数据", default=1, type=int)
    return parser.parse_args()


def split_data(data, row=0):
    '''
    根据行数row对输入数据data进行切分。步长为4，起始值为实际文件行数-1
    返回取出的信号
    '''
    data = data[row:45000*4:4]
    return data


def read_data(filename):
    '''
    读文件并转换为list(float)，返回数据list
    '''
    data = open(filename).read()
    data = data.split( )
    data = [float(s) for s in data]
    # s1 = data[0:45000*4:4]
    return data


def bw_filter(data, fs=3000, lowcut=1, highcut=30, order=2):
    '''
    对list型输入data进行butterworth滤波，返回滤波后的数据
    '''
    nyq = 0.5*fs  # nyquist frequency of sample
    low = lowcut/nyq
    high = highcut/nyq
    b,a = signal.butter(order, [low,high], btype='bandpass')
    data = signal.lfilter(b, a, data)
    return data


def draw(s1, s1f, figdir):
    '''
    对未滤波信号s1和已滤波信号s1f进行绘图，保存在figdir路径
    '''
    fig = plt.figure()

    ax1 = fig.add_subplot(211)
    plt.plot(s1,color='r')
    ax1.set_title('Original Signal')
    plt.ylabel('Amplitude')

    ax2 = fig.add_subplot(212)
    plt.plot(s1f,color='r')
    ax2.set_title('Denoised Signal')
    plt.ylabel('Amplitude')


    plt.savefig(figdir)
    plt.close('all')


def preprocessz(X_all):
    '''
    对X_all数组进行归一化
    '''
    X_all = X_all - np.mean(X_all)
    X_all = X_all / np.ptp(X_all)
    return X_all


def preprocess(file, row):
    '''
    输入指定file文件和指定行row，进行全部预处理过程
    先读取文件、再选择指定行、再进行滤波
    以ndarray返回原信号和滤波后的信号
    '''
    filename = "./data/20151026_" + file
    data = read_data(filename)
    raw = np.array(split_data(data, row=row))
    raw = preprocessz(raw)
    filtered = bw_filter(raw)
    return raw, filtered


'''
当本文件作为主函数执行时：滤波并将滤波前后的信号对比，并作图
'''
if __name__ == "__main__":
    args = arg_parse()  # 获取参数
    raw, filtered = preprocess(args.filename, args.row-1)  # 预处理
    if not os.path.exists(os.path.dirname("./figures/filter")):  # 创建目录
        os.mkdir("./figures/filter")
    figdir = "./figures/filter/" + args.filename + "_" + str(args.row) + ".png"  # 设置保存路径
    draw(raw, filtered, figdir)  # 画图

'''
这里对butterworth低通滤波后的数据进行了快速Fourier变换，并打印频谱，但打印出的频谱中显示基本全是高频分量。
这一现象非常反常，需要后续研究。
'''
    # from scipy.fft import fft
    # ff_train = np.array(fft(filtered), dtype=complex)
    # freq0 = np.array([np.angle(ff_train),np.abs(ff_train)])
    # g = freq0[1]
    # g = g[int(len(g)/2):-1]
    # fig = plt.figure()
    # plt.plot(g, color='blue')
    # plt.show()