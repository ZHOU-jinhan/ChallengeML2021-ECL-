import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import h5py

class SVM:
    def __init__(self):
        self.model=None
        
    def fit(self,path_x='X_train.h5',path_y='Y_train.csv'):
        h5_file=h5py.File(path_x,"r");X_data=(np.array(h5_file["data"][:, 2:])).astype('float32')
        X_train = X_data.reshape(X_data.shape[0],8,-1,100)
        X_train = X_train.transpose(0,2,3,1);X=X_train.reshape(-1,8);
        self.mean = np.mean(X,axis=0);self.aplt = np.max(X,axis=0)-np.min(X,axis=0)
        X_train = (X_train-self.mean)/self.aplt
        X_train = X_train.reshape(X_train.shape[0]*X_train.shape[1],-1)
        Y_train = pd.read_csv(path_y, header=0, index_col=0).values.astype('float32').reshape(-1,)
        print(X_train.shape,Y_train.shape)
        self.model = svm.SVC(C=0.75, kernel='rbf', gamma='auto', decision_function_shape='ovr', cache_size=500)
        self.model.fit(X_train, Y_train)
        self.save()

    def predict(self,path_x='X_train.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        test_X = test_X.reshape(test_X.shape[0],8,-1,100)
        test_X = test_X.transpose(0,2,3,1)
        test_X = (test_X-self.mean)/self.aplt
        test_X = test_X.reshape(test_X.shape[0]*test_X.shape[1],-1)
        test_y = self.model.predict(test_X)
        if path_x=="X_train.h5":
            pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("train/LSTM_y.csv")
        else:
            pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("results/test_y/LSTM_y.csv")

    def save(self):
        np.save("results/models/SVM/mean.npy",self.mean)
        np.save("results/models/SVM/aplt.npy",self.aplt)
        joblib.dump(self.model, '"results/models/SVM/SVM.model')

    def load(self):
        self.mean=np.load("results/models/SVM/mean.npy")
        self.aplt=np.load("results/models/SVM/aplt.npy")
        joblib.load(model, 'results/models/SVM/SVM.model')

if __name__ == '__main__':
    mySVM=SVM()
    mySVM.fit()
    mySVM.predict()
    
