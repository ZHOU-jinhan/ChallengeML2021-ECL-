import h5py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class myLinear:
    def __init__(self):
        self.model=None

    def train(self,path_x='X_train.h5',path_y='Y_train.csv',train_split=1):
        # load dataset
        h5_file=h5py.File(path_x,"r")
        X_train=(np.array(h5_file["data"][:, 2:])).astype('float32')
        X_train = X_train.reshape(X_train.shape[0],8,int(X_train.shape[1]/800),100)
        X_train = X_train.transpose(0,2,1,3)
        X_train = np.mean(X_train,3)
        X_train = X_train.reshape(X_train.shape[0],-1)
        Y_train = pd.read_csv(path_y, header=0, index_col=0).values.astype('float32')
        # split into train and test sets
        n_train_hours = int(X_train.shape[0]*train_split)
        # split into input and outputs
        train_X, train_y = X_train[:n_train_hours,:], Y_train[:n_train_hours, :]
        test_X, test_y = X_train[n_train_hours:,:], Y_train[n_train_hours:, :]
        X=np.matrix(train_X);Y=np.matrix(train_y)
        print((X.T*X).shape,(X.T*Y).shape)
        self.theda=np.linalg.inv(X.T*X)*X.T*Y
        if train_split==1:
            test_X, test_y = train_X, train_y
        print(train_X.shape,train_y.shape,self.theda.shape)
        yhat = np.array(np.matrix(test_X)*self.theda)
        self.d_threshold=[]
        for i in range(test_y.shape[1]):
            if len(np.where(test_y[:,i]==1)[0])==0:
                self.d_threshold.append(1.05*np.max(yhat[np.where(test_y[:,i]==0)[0],i],0))
            else:
                self.d_threshold.append(0.5*np.mean(yhat[np.where(test_y[:,i]==1)[0],i],0)+0.5*np.mean(yhat[np.where(test_y[:,i]==0)[0],i],0))
        self.d_threshold=np.array(self.d_threshold).astype("float32")
        # calculate RMSE
        rmse = np.sqrt(mean_squared_error(test_y, yhat))
        print('Test RMSE: %.3f' % rmse)
        print(yhat.shape,self.d_threshold.shape,(yhat>self.d_threshold).shape)
        rmse = np.sqrt(mean_squared_error(test_y, (yhat>self.d_threshold).astype("int32")))
        print('Test RMSE(0/1): %.3f' % rmse)
        self.save()

    def predict(self,path_x='X_test.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        test_X = test_X.reshape(test_X.shape[0],8,int(test_X.shape[1]/800),100)
        test_X = test_X.transpose(0,2,1,3)
        test_X = np.mean(test_X,3)
        test_X = test_X.reshape(test_X.shape[0],-1)
        test_y = np.array(np.matrix(test_X)*self.theda)
        pd.DataFrame(test_y).to_csv("results/y_hat/Linear_y_hat.csv")
        pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("results/test_y/Linear_y.csv")

    def save(self):
        """
        Save trained model.
        """
        np.save("results/models/Linear/threshold.npy",self.d_threshold)
        np.save('results/models/Linear/linear_.npy',np.array(self.theda))

    def load(self):
        """
        Load model for channel.
        """
        self.d_threshold=np.load("results/models/Linear/threshold.npy")
        self.theda = np.matrix(np.load('results/models/Linear/linear_.npy'))
        
if __name__=="__main__":
    linear=myLinear()
    linear.load()
    linear.predict()
