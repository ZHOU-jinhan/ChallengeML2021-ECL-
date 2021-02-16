import pandas as pd
import os
Y_train=pd.read_csv(r"y_train.csv", header=0, index_col=0)
col=Y_train.columns;ind=Y_train.index[-1]+1
for f in os.listdir(r"results/test_y"):
    Y = pd.DataFrame(pd.read_csv(r"results/test_y/"+f, header=0, index_col=0).values.astype('float32'),columns=Y_train.columns)
    Y.rename(index={i:i+ind for i in Y.index},inplace=True)
    Y.to_csv(r"results/test_y/"+f)
for f in os.listdir(r"results/y_hat"):
    Y = pd.DataFrame(pd.read_csv(r"results/y_hat/"+f, header=0, index_col=0).values.astype('float32'),columns=Y_train.columns)
    Y.rename(index={i:i+ind for i in Y.index},inplace=True)
    Y.to_csv(r"results/y_hat/"+f)
