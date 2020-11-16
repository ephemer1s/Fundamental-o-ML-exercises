import pandas
import pickle
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import matplotlib.pyplot as plt
import argparse

def arg_parse():
    '''
    获取运行参数，返回一个可被调用的参数对象
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--model","-m", dest="model"
                        , help="选择预测模型", default="adaboost", type=str)
    return parser.parse_args()

if __name__ == "__main__":
# 取得运行参数
    args = arg_parse()
    model = "./pickles" + args.model + ".pickle"
    with open(model, 'rb') as f:
        classifier = pickle.load(f)  # 加载训练好的决策树模型


# 加载测试数据
    filename = "./data/test.csv"
    data = pandas.read_csv(filename,index_col=0)
    X = data.iloc[:,0:9]  # 测试集属性
    y_true = data['label']  # 真实标签
    
# 进行测试
    y_pred = classifier.predict(X)  # 测试决策树
    confusion = confusion_matrix(y_true, y_pred)  # 得到混淆矩阵

    print(f1_score(y_true, y_pred, average="macro"))  # 得到测试结果F1分数
    print(f1_score(y_true, y_pred, average="micro"))
    print(f1_score(y_true, y_pred, average="weighted"))

# 画图
    figure, ax = plt.subplots()
    # 画混淆矩阵    
    plt.imshow(confusion, cmap=plt.cm.Blues)
    # 画色卡
    plt.colorbar()     
    # 画坐标轴刻度
    y_tick = ['full peaks', 'half peak', 'no peak']
    classes = list(set(y_tick))
    indices = range(len(confusion))
    plt.tick_params(labelsize=11)  
    plt.xticks(indices, classes)
    plt.yticks(indices, classes, rotation=90)
    # 画坐标轴标题
    font1 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 15,}
    plt.xlabel('prediction', font1)
    plt.ylabel('real label', font1)
    # 画图的标题
    font2 = {'family' : 'Times New Roman','weight' : 'normal','size'   : 23,}
    plt.title('Confusion Matrix', font2)
    # 画混淆矩阵上的文字标注
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    # 保存
    savedest = "./figures/ConfusionMatrix_" + args.model + ".png"
    plt.savefig(savedest)