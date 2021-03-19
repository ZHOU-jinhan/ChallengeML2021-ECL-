import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

path_x='X_train.h5';
path_y='Y_train.csv'
passbatch_size=80
# load dataset
h5_file=h5py.File(path_x,"r")
X_train=(np.array(h5_file["data"][:, 2:])).astype('float32')
Y_train = pd.read_csv(path_y, header=0, index_col=0).values.astype('float32')
path_x='X_test.h5'
h5_file = h5py.File(path_x,"r")
test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')


x = np.vstack((X_train,test_X ))
x = PCA(n_components=20).fit_transform(x)
x = TSNE(n_components=2).fit_transform(x)
X_train= x[:4400,:]
test_X = x[4400:,:]


clf = KNeighborsClassifier(n_neighbors=7)
clf.fit(X_train,Y_train)
test_y = clf.predict(test_X)
test_y = test_y.reshape(4400,-1)

pd.DataFrame(test_y).to_csv("results/y_hat/KNN_y_FIN.csv")

Y_benchmark=pd.read_csv('y_benchmark.csv', header=0, index_col=0)
Y = pd.DataFrame(pd.read_csv('results/y_hat/KNN_y_FIN.csv',
                             header=0, index_col=0).values.astype('int32'),
                            columns=Y_benchmark.columns)

Y.rename(index={i:i+4400 for i in Y.index},inplace=True) # index即第一列改为和benchmark一样，从4400开始
Y.to_csv('results/y_hat/KNN_y_FIN.csv')