from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential,load_model,Model
from tensorflow.keras.layers import Input,BatchNormalization,LeakyReLU,Dense,LSTM,Dropout,Activation,Reshape,Conv2D,UpSampling2D,MaxPooling2D
import numpy as np
import h5py
import os
import pandas as pd

class myCRGAN:
    def __init__(self):
        self.model=None

    def generat(self):
        model = Sequential()
        model.add(Dense(self.input_shape[0] * self.input_shape[1]*self.input_shape[2], activation="relu",input_dim=self.latent_dim))
        model.add(Reshape(self.input_shape))
        model.add(UpSampling2D())
        # Convolution layer
        model.add(Conv2D(32, (2, 5), padding='same', kernel_initializer='he_normal'))  # (None, ls, 38, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(MaxPooling2D(pool_size=(1, 2), name='max1'))  # (None, ls/2, 19, 64)
        model.add(Conv2D(32, (5, 2), padding='same', kernel_initializer='he_normal'))  # (None, ls/2, 19, 64)
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        # CNN to RNN
        model.add(Reshape(target_shape=((self.input_shape[0], -1))))  # (None, ls, -1)
        model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))  # (None, ls, 32)
        # RNN layer
        model.add(LSTM(80, input_shape=(None, 32), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Dense(self.input_shape[1]*self.input_shape[2], activation='tanh'))
        model.add(Reshape(self.input_shape))
        model.summary()
        noise = Input(shape=(self.latent_dim,))
        gen_sample = model(noise)
        return Model(noise, gen_sample)

    def discriminat(self):
        model = Sequential()
        # Convolution layer
        model.add(Conv2D(16, (3, 3), input_shape=self.input_shape, padding='same', kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(1, 2)))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (2, 5), padding='same', kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (5, 2), padding='same', kernel_initializer='he_normal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        # CNN to RNN
        model.add(Reshape(target_shape=((90,-1)))) 
        model.add(Dense(128))  # (None, ls, 32)
        model.add(Dropout(0.25))
        model.add(Dense(64))
        # CNN to RNN
        model.add(LSTM(80, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(40, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(Dense(1,activation="sigmoid"))
        model.compile(loss='mae', optimizer='adam')
        model.summary()
        sample = Input(shape=self.input_shape)
        validity = model(sample)
        return Model(sample, validity)

    def train(self,path_x='X_train.h5',path_y='Y_train.csv',epochs=50,batch_size=80,train_split=1,plot=1):
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
        if self.model==None:
            self.input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # (ls, 38, 1)
            self.latent_dim=X_train.shape[1]
            # Build and compile the discriminator
            self.discriminator = self.discriminat()
            self.discriminator.compile(loss='mae',optimizer='adam',metrics=['accuracy'])
            # Build the generator
            self.generator = self.generat()
            # The generator takes noise as input and generates imgs
            z = Input(shape=(90,))
            sample = self.generator(z)
            # For the combined model we will only train the generator
            self.discriminator.trainable = False
            # The discriminator takes generated images as input and determines validity
            valid = self.discriminator(sample)
            # The combined model  (stacked generator and discriminator)
            # Trains the generator to fool the discriminator
            self.combined = Model(z, valid)
            self.combined.compile(loss='mae',optimizer='adam',metrics=['accuracy'])
        # Adversarial ground truths
        fake = np.ones((batch_size, 90))
        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, train_X.shape[0],batch_size)
            samples = train_X[idx]
            valid = train_y[idx]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Generate a batch of new images
            gen_samples = self.generator.predict(noise)
            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(samples, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_samples, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)
            # ---------------------
            #  claculate the aggregative indicator
            # ---------------------
            acc = 100 * d_loss[1] # the accuracy metric of discriminator
            # Plot the progress
            print("epoch:"+str(epoch)+"||D loss:"+str(round(d_loss[0],4))+"||G loss:"+str(round(g_loss[0],4))+"||D accuracy:"+str(round(acc,4))+"%")
            # judge whether reach to epochs and whether to early stop
            # if acc>99.95 and d_loss[0]<0.002:
            #    break
        if train_split==1:
            test_X=train_X;test_y=train_y
        # make a prediction
        yhat = self.discriminator.predict(test_X)
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

    def save(self):
        """
        Save trained model.
        """
        np.save(r"results/models/CRGAN/threshold.npy",self.d_threshold)
        self.combined.save(os.path.join('results' ,'models','CRGAN','combined_model.h5'))
        self.discriminator.save(os.path.join('results', 'models','CRGAN','discriminator.h5'))
        self.generator.save(os.path.join('results', 'models','CRGAN','generator.h5'))

    def load(self):
        """
        Load model for channel.
        """
        self.d_threshold=np.load(r"results/models/CRGAN/threshold.npy")
        self.discriminator = load_model(os.path.join('results','models','CRGAN', 'discriminator.h5'))
        try:
            self.combined = load_model(os.path.join('results','models','CRGAN', 'combined_model.h5'))
            self.generator = load_model(os.path.join('results','models','CRGAN', 'generator.h5'))
        except:
            pass
        self.discriminator.summary()

    def predict(self,path_x='X_test.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        test_X = test_X.reshape(test_X.shape[0],8,int(test_X.shape[1]/800),100)
        test_X = test_X.transpose(0,2,1,3)
        test_y = self.discriminator.predict(test_X)
        test_y = test_y.reshape(test_y.shape[0],test_y.shape[1])
        pd.DataFrame(test_y).to_csv("results/y_hat/CRGAN_y_hat.csv")
        pd.DataFrame((test_y>self.d_threshold).astype("int32")).to_csv("results/test_y/CRGAN_y.csv")

if __name__=="__main__":
    crgan=myCRGAN()
    crgan.load()
    crgan.predict()
