import numpy as np
from numpy import mean, median, std, var, amax, amin, percentile
from scipy.stats import kurtosis
from scipy.signal import find_peaks
import os
import shutil
import argparse
import matplotlib.pyplot as plt
import pandas

# 引用了butterworth.py中的preprocess函数。
from butterworth import preprocess


def arg_parse():
    '''
    获取运行参数，返回一个可被调用的参数对象
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--filename","-f", dest='filename'
                        , help="要读取的文件序号", default="113", type=str)
    parser.add_argument("--row","-r", dest='row'
                        , help="选择第几行数据", default=1, type=int)
    parser.add_argument("--length","-l", dest='slidelength'
                        , help="切片长度", default=150, type=int)
    parser.add_argument("--draw","-d", dest='willdraw'
                        , help="是否画图", default=1, type=int)
    return parser.parse_args()


def split_signal(sig, length=150):
    '''
    输入原长45000的信号sig，对其按length长度切片，返回切片后的数据，为ndarray
    '''
    if type(sig) == "list":
        sig = np.array(sig)
    sig = sig.reshape((int(45000/length)), length, order="A")
    return sig


def draw(data, num):
    '''
    输入切片后的信号data和切片序号num，画图并保存
    '''
    fig = plt.figure()
    if not os.path.exists(os.path.dirname("./data/separate")):
        os.mkdir("./data/separate")
    figdir = "./data/separate/" + str(num) + ".png"
    plt.plot(data, color="b")
    plt.savefig(figdir)
    plt.close('all')


def calc_attr(data, add_label=True):
    '''
    输入切片信号data，对其求统计特征量，返回包含如下统计特征量的list。
    如果add_label为true，则自动给数据在最后一行添加label。
    '''
    attr = []
    
    attr.append(amax(data))  # 最大值
    attr.append(percentile(data, 75))  # 上四分位
    attr.append(median(data))  # 中位数
    attr.append(percentile(data, 25))  # 下四分位
    attr.append(amin(data))  # 最小值

    attr.append(mean(data))  # 均值
    attr.append(std(data))  # 标准差
    attr.append(var(data))  # 方差
    attr.append(kurtosis(data))  # 峰度

    peaks, _ = find_peaks(data)  # 波峰位置
    attr.append(len(peaks))  # 波峰个数

    if add_label:
        # label = amax(data) - amin(data)
        # label = int(label + 0.5)
        label = (amax(data) - amin(data) > 0.1) and 1
        label = int(label)
        if label == 1:
            if amax(data) == data[0] or amax(data) == data[-1]:
                label = 2
        attr.append(label)

    return attr


def removedir(folder):
    '''
    删除folder目录下的所有文件
    '''
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


if __name__ == "__main__": 
# 获取参数
    args = arg_parse()

# 调用butterworth.py中的预处理函数，对数据进行预处理
    raw, filtered = preprocess(args.filename, args.row-1)  # 预处理，得到未滤波和滤波的信号
    raw = np.array(raw)  # 转为np格式
    print(raw.shape)  

# 切片
    raw = split_signal(raw, length=args.slidelength)  # 未处理数据切片
    filtered = split_signal(filtered, length=args.slidelength)  # 滤波后数据切片

# 计算数据属性
    list_all = []  # 声明属性表容器，其中的每个元素也为list
    for data in filtered:
        list_all.append(calc_attr(data))  # 计算每段切片信号的特征，将得到的list添加到总list中

# 用pandas写入csv文件
    filename = "./data/" + str(args.filename) + "_" + str(args.row) + ".csv"
    save = pandas.DataFrame(list_all, columns=['max', 'uquartile', 'median', 'lquartile', 'min', 'mean', 'std', 'var', 'kurtosis', 'peaks', 'label'])
    save.to_csv(filename)

# 对分段数据画图
    if args.willdraw:
        removedir("./figures/separate/")  # 清空保存信号段文件夹
        for i in range(filtered.shape[0]): 
            draw(filtered[i], i)  # 对每个信号段绘图


    