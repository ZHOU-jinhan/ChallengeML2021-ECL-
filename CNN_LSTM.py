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
from tensorflow.keras.layers import BatchNormalization,LeakyReLU,Dense,LSTM,Dropout,Conv2D,MaxPooling2D,Reshape
import h5py
import numpy as np
import pandas as pd

class myCNN_LSTM:
    def __init__(self):
        self.model=None

    def train(self,path_x='X_train.h5',path_y='Y_train.csv',epochs=30,batch_size=80,train_split=1,plot=1):
        # load dataset
        h5_file = h5py.File(path_x,"r");X_data=(np.array(h5_file["data"][:, 2:])).astype('float32')
        X_train = X_data.reshape(X_data.shape[0],8,int(X_data.shape[1]/800),100)
        X_train = X_train.transpose(0,2,1,3)
        Y_train = read_csv(path_y, header=0, index_col=0).values.astype('float32')
        # split into train and test sets
        n_train_hours = int(X_train.shape[0]*train_split)
        # split into input and outputs
        train_X, train_y = X_train[:n_train_hours,:,:], Y_train[:n_train_hours, :]
        if train_split!=1:
            test_X, test_y = X_train[n_train_hours:,:,:], Y_train[n_train_hours:, :]
        print(train_X.shape,train_y.shape)
        # design network
        if self.model==None:
            self.model = Sequential()
            # Convolution layer
            self.model.add(Conv2D(16,[3,3], input_shape=(train_X.shape[-3],train_X.shape[-2], train_X.shape[-1]), padding='same', kernel_initializer='he_normal'))  # (None, ls, 38, 32)
            self.model.add(MaxPooling2D([1,2],padding="same"))
            self.model.add(BatchNormalization(momentum=0.8))
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Dropout(0.25))
            self.model.add(Conv2D(32, [2,5], padding='same', kernel_initializer='he_normal'))  # (None, ls, 38, 64)
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Dropout(0.25))
            self.model.add(Conv2D(32, [5,2], padding='same', kernel_initializer='he_normal'))  # (None, ls/2, 19, 64)
            self.model.add(LeakyReLU(alpha=0.2))
            self.model.add(Dropout(0.25))
            self.model.add(Reshape(target_shape=((90,-1))))
            self.model.add(Dense(128))
            self.model.add(Dropout(0.25))
            self.model.add(Dense(64))
            # CNN to RNN
            self.model.add(LSTM(80, return_sequences=True))
            self.model.add(Dropout(0.3))
            self.model.add(LSTM(40, return_sequences=True))
            self.model.add(Dropout(0.3))
            self.model.add(Dense(1,activation="sigmoid"))
            self.model.compile(loss='mae', optimizer='adam')
            self.model.summary()
        # fit network
        if train_split!=1:
            history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        else:
            history = self.model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(train_X, train_y), verbose=2, shuffle=False)
        # plot history
        pyplot.plot(history.history['loss'], label='train')
        if train_split!=1:
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
        for i in range(test_y.shape[1]):
            if len(np.where(test_y[:,i]==1)[0])==0:
                self.d_threshold.append(1.05*np.max(yhat[np.where(test_y[:,i]==0)[0],i],0))
            else:
                self.d_threshold.append(0.5*np.mean(yhat[np.where(test_y[:,i]==1)[0],i],0)+0.5*np.mean(yhat[np.where(test_y[:,i]==0)[0],i],0))
        self.d_threshold=np.array(self.d_threshold).astype("float32")
        # calculate RMSE
        rmse = sqrt(mean_squared_error(test_y, yhat))
        print('Test RMSE: %.3f' % rmse)
        rmse = sqrt(mean_squared_error(test_y, np.array(yhat>self.d_threshold).astype("int32")))
        print('Test RMSE(0/1): %.3f' % rmse)
        self.save()
        if plot:
            pyplot.show()

    def predict(self,path_x='X_test.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        test_X = test_X.reshape(test_X.shape[0],8,int(test_X.shape[1]/800),100)
        test_X = test_X.transpose(0,2,1,3)
        test_y = self.model.predict(test_X)
        test_y = test_y.reshape(test_y.shape[0],test_y.shape[1])
        pd.DataFrame(test_y).to_csv("results/y_hat/CRNN_y_hat.csv")
        pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("results/test_y/CRNN_y.csv")

    def save(self):
        """
        Save trained self.model.
        """
        np.save("results/models/CRNN/threshold.npy",self.d_threshold)
        self.model.save('results/models/CRNN/cnn_lstm_.h5')

    def load(self):
        """
        Load self.model for channel.
        """
        self.d_threshold=np.load("results/models/CRNN/threshold.npy")
        self.model = load_model('results/models/CRNN/cnn_lstm_.h5')
        self.model.summary()

if __name__=="__main__":
    cnn_lstm=myCNN_LSTM()
    cnn_lstm.load()
    cnn_lstm.predict()
