import pandas

df1 = pandas.read_csv("113_1.csv",index_col=0)
df2 = pandas.read_csv("113_2.csv",index_col=0)
df3 = pandas.read_csv("113_3.csv",index_col=0)
df4 = pandas.read_csv("113_4.csv",index_col=0)
frames = [df1, df2, df3, df4]
frames = pandas.concat(frames)
frames.to_csv("./data/train.csv")

df1 = pandas.read_csv("114_1.csv",index_col=0)
df2 = pandas.read_csv("114_2.csv",index_col=0)
df3 = pandas.read_csv("114_3.csv",index_col=0)
df4 = pandas.read_csv("114_4.csv",index_col=0)
frames = [df1, df2, df3, df4]
frames = pandas.concat(frames)
frames.to_csv("./data/test.csv")