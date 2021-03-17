from math import sqrt
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Input,Attention,BatchNormalization,Dense,LSTM,Dropout,Conv2D,AveragePooling2D,Reshape
import tensorflow as tf
import numpy as np
import pandas as pd
import h5py

opt = [tf.keras.optimizers.Adam(),tf.keras.optimizers.SGD(momentum=0.2,nesterov=True)]

class myAttention_CNN_LSTM:
    def __init__(self):
        self.model=None

    def train(self,path_x='X_train.h5',path_y='Y_train.csv',epochs=30,batch_size=80,train_split=1,plot=1):
        # load dataset
        h5_file = h5py.File(path_x,"r");X_data=(np.array(h5_file["data"][:, 2:])).astype('float32')
        X_train = X_data.reshape(X_data.shape[0],8,-1,100)
        X_train = X_train.transpose(0,2,3,1);X=X_train.reshape(-1,8);
        self.mean = np.mean(X,axis=0);self.aplt = np.max(X,axis=0)-np.min(X,axis=0)
        X_train = (X_train-self.mean)/self.aplt
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],-1,1)
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
            input_ = Input(shape=(train_X.shape[-3:]))
            output_= Attention()([input_,input_,input_])
            # Convolution layer
            output_=Conv2D(16,[3,5], input_shape=(train_X.shape[-3],train_X.shape[-2], train_X.shape[-1]), padding='same')(output_)
            output_=AveragePooling2D([1,5],padding="same")(output_)
            output_=BatchNormalization(momentum=0.8)(output_)
            output_=Conv2D(32, [3,5],padding='same')(output_)
            output_=Conv2D(32, [3,5], padding='same')(output_)
            output_=AveragePooling2D([1,5],padding="same")(output_)
            output_=Dropout(0.25)(output_)
            output_=Reshape(target_shape=((90,-1)))(output_)
            # CNN to RNN
            output_=LSTM(256, return_sequences=True)(output_)
            output_=Dropout(0.3)(output_)
            output_=LSTM(64, return_sequences=True)(output_)
            output_=Dropout(0.3)(output_)
            output_=Dense(1,activation="sigmoid")(output_)
            self.model=Model(input_,output_)
            self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='adam')
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
        rmse = sqrt(mean_squared_error(test_y, (yhat>self.d_threshold).astype("int32")))
        print('Test RMSE(0/1): %.3f' % rmse)
        self.save()
        if plot:
            pyplot.show()
        
    def predict(self,path_x='X_test.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        test_X = test_X.reshape(test_X.shape[0],8,-1,100)
        test_X = test_X.transpose(0,2,3,1);test_X = (test_X-self.mean)/self.aplt
        test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],-1,1)
        test_y = self.model.predict(test_X)
        test_y = test_y.reshape(test_y.shape[0],test_y.shape[1])
        pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("train/attCRNN_y.csv")
        # pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("results/test_y/attCRNN_y.csv")

    def save(self):
        """
        Save trained self.model.
        """
        np.save("results/models/attCRNN/threshold.npy",self.d_threshold)
        self.model.save('results/models/attCRNN/att_cnn_lstm_.h5')
        np.save("results/models/attCRNN/mean.npy",self.mean)
        np.save("results/models/attCRNN/aplt.npy",self.aplt)

    def config(self,i=1):
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt[i])

    def load(self):
        """
        Load self.model for channel.
        """
        self.mean=np.load("results/models/attCRNN/mean.npy")
        self.aplt=np.load("results/models/attCRNN/aplt.npy")
        self.d_threshold=np.load("results/models/attCRNN/threshold.npy")
        self.model = load_model('results/models/attCRNN/att_cnn_lstm_.h5')
        self.model.summary()

if __name__=="__main__":
    attention_cnn_lstm=myAttention_CNN_LSTM()
    attention_cnn_lstm.train()
    attention_cnn_lstm.predict()
