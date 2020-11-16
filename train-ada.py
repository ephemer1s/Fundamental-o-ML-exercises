from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score

import argparse
import pandas
import pickle

import numpy as np
import matplotlib.pyplot as plt


def arg_parse():
    '''
    获取运行参数，返回一个可被调用的参数对象
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument("--num","-n", dest='num_weakdt'
                        , help="弱分类器数量", default=20, type=int)
    parser.add_argument("--lr","-lr", dest='lr'
                        , help="学习率", default=0.5, type=float)
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parse()
# 加载训练数据
    filename = "./data/train.csv"
    data = pandas.read_csv(filename,index_col=0)
    X = data.iloc[:,0:9]  # 前十行为属性
    y = data['label']  # 最后一行为标签
# 构造adaboost分类器
    ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=1
        ),
        n_estimators=args.num_weakdt, 
        learning_rate=args.lr,
        random_state=0
    )
    filename = "train.csv"
    data = pandas.read_csv(filename,index_col=0)
    X1 = data.iloc[:,0:9]  # 测试集属性
    y1_true = data['label']  # 真实标签
# 训练adaboost
    ada.fit(X, y)
    with open('./pickles/adaboost.pickle', 'wb') as f:
        pickle.dump(ada, f)  # 保存训练好的决策树模型

    # start = 10
    # stop = 100
    # padding = 1
    # learningrates = np.linspace(start, stop, int((stop-start)/padding))
    # learningrates = [int(i) for i in learningrates]
    # scores = []
    # bestscore = 0
    # for i in learningrates:
    #     ada.set_params(n_estimators=i)
    #     ada.fit(X, y)
    #     y1_pred = ada.predict(X1)  # 测试决策树
    #     score = (f1_score(y1_true, y1_pred, average="macro"))  # 得到测试结果F1分数
    #     bestscore = max(score, bestscore)
    #     scores.append(score)

    # plt.plot(learningrates, scores, color="red")
    # plt.show()


    