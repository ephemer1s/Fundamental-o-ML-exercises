from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas
import pickle


if __name__ == "__main__":
# 加载训练数据
    filename = "train.csv"
    data = pandas.read_csv(filename,index_col=0)
    X = data.iloc[:,0:9]  # 前十行为属性
    y = data['label']  # 最后一行为标签

# 训练决策树
    dtc = DecisionTreeClassifier(  # 构造决策树
        criterion="gini", splitter="best")
    dtc.fit(X, y)  # 训练决策树

    with open('./pickles/tree.pickle', 'wb') as f:
        pickle.dump(dtc, f)  # 保存训练好的决策树模型

    visname = './figures/tree.dot'
    with open(visname, 'w') as f:  # 保存决策树的可视化dot文件
        f = export_graphviz(dtc, feature_names=X.columns, out_file=f)
    