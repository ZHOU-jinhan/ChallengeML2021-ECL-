from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import BatchNormalization,Dense,LSTM,Dropout,Reshape,Permute
import tensorflow as tf
import h5py
import numpy as np
import pandas as pd

class myLSTM:
    def __init__(self):
        self.model=None

    def train(self,path_x='X_train.h5',path_y='Y_train.csv',epochs=30,batch_size=80,train_split=1,plot=1):
        # load dataset
        h5_file=h5py.File(path_x,"r");X_data=(np.array(h5_file["data"][:, 2:])).astype('float32')
        print(X_data.shape)
        X_train = X_data.reshape([X_data.shape[0],8,-1,100])
        X_train = X_train.transpose(0,2,3,1);X=X_train.reshape([-1,X_train.shape[-1]]);
        self.mean = np.mean(X,axis=0);self.aplt = np.max(X,axis=0)-np.min(X,axis=0)
        X_train = (X_train-self.mean)/self.aplt
        X_train = X_train.transpose(0,3,1,2)
        Y_train = read_csv(path_y, header=0, index_col=0).values.astype('float32')
        # split into train and test sets
        n_train_hours = int(X_train.shape[0]*train_split)
        # split into input and outputs
        train_X, train_y = X_train[:n_train_hours], Y_train[:n_train_hours]
        if train_split!=1:
            test_X, test_y = X_train[n_train_hours:], Y_train[n_train_hours:]
##        id_ = np.where(np.sum(train_y,axis=1)>0)[0]
##        for _ in range(3):
##            train_X = np.append(train_X,train_X[id_],axis=0)
##            train_y = np.append(train_y,train_y[id_],axis=0)
        print(train_X.shape,train_y.shape)
        # design network
        if self.model==None:
            self.model = Sequential()
            self.model.add(Dense(10,input_shape=X_train.shape[1:]))
            self.model.add(BatchNormalization(momentum=0.5))
            self.model.add(Reshape(target_shape=(8,-1)))
            self.model.add(Permute((2,1)))
            self.model.add(LSTM(40, return_sequences=True))
            self.model.add(Dropout(0.25))
            self.model.add(LSTM(40, return_sequences=False))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(90,activation="softmax"))
            self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(lr=2e-3))
        # fit network
        if train_split!=1:
            history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        else:
            history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(train_X, train_y), verbose=2, shuffle=False)
        self.model.summary()
        # plot history
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        if train_split==1:
            test_X=train_X;test_y=train_y
        # make a prediction
        yhat = self.model.predict(test_X)
        yhat = yhat.reshape(yhat.shape[0],yhat.shape[1])
        # invert scaling for forecast
        # invert scaling for actual
        test_y = test_y.reshape(yhat.shape[0],yhat.shape[1])
        self.d_threshold=[]
        if np.where(test_y==0)[0].size==0:
            self.d_threshold.append(np.max(yhat[np.where(test_y==0)]))
        else:
            print(np.min(yhat[np.where(test_y==1)]),np.mean(yhat[np.where(test_y==1)]),np.max(yhat[np.where(test_y==1)]),\
                  np.min(yhat[np.where(test_y==0)]),np.mean(yhat[np.where(test_y==0)]),np.max(yhat[np.where(test_y==0)]))
            self.d_threshold.append(0.5*np.mean(yhat[np.where(test_y==1)])\
                                    +0.5*np.mean(yhat[np.where(test_y==0)]))
        self.d_threshold=np.array(self.d_threshold).astype("float32")
        # calculate RMSE
        rmse = sqrt(mean_squared_error(test_y, yhat))
        print('Test RMSE: %.3f' % rmse)
        rmse = sqrt(mean_squared_error(test_y, (yhat>self.d_threshold).astype("int32")))
        print('Test RMSE(0/1): %.3f' % rmse)
        self.save()
        if plot:
            pyplot.show()

    def predict(self,path_x='X_train.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        test_X = test_X.reshape(test_X.shape[0],8,-1,100)
        test_X = test_X.transpose(0,2,3,1)
        test_X = (test_X-self.mean)/self.aplt
        test_X = test_X.transpose(0,3,1,2)
        test_y = self.model.predict(test_X)
        if path_x=="X_train.h5":
            pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("train/LSTM_y.csv")
        else:
            pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("results/test_y/LSTM_y.csv")

    def save(self):
        """
        Save trained model.
        """
        np.save("results/models/LSTM/threshold.npy",self.d_threshold)
        np.save("results/models/LSTM/mean.npy",self.mean)
        np.save("results/models/LSTM/aplt.npy",self.aplt)
        self.model.save('results/models/LSTM/lstm_.h5')

    def load(self):
        """
        Load model for channel.
        """
        self.d_threshold=np.load("results/models/LSTM/threshold.npy")
        self.mean=np.load("results/models/LSTM/mean.npy")
        self.aplt=np.load("results/models/LSTM/aplt.npy")
        self.model = load_model('results/models/LSTM/lstm_.h5')
        self.model.summary()

if __name__=="__main__":
    lstm=myLSTM()
##    lstm.load()
    lstm.train()
    lstm.predict()
