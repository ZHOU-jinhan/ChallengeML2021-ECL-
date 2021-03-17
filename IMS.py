import pandas as pd
import numpy as np
import h5py
import os
from sklearn.metrics import mean_squared_error
from time import time

class uniIMS:
    def __init__(self):
        # IMS参数
        # 簇外的点是否需要通过调整簇的大小，从而将其纳入簇内的门限
        self.incorporate_dist_threshold = 0.3
        # 初始簇大小
        self.initial_dist_threshold = 0.1
        # 簇的增长大小，对于某个维度的x_max和次大的x_submax的差的5%增长在x_max上,x_min同理
        self.expansion_rate = 0.02

    def fit(self,X_train):
        t=time()
        self.mean=np.mean(X_train,axis=0);self.alt=np.max(X_train,axis=0)-np.min(X_train,axis=0)
        self.alt[np.where(self.alt==0)[0]]=0.01
        X=(X_train-self.mean)/self.alt
        # 生成第一个簇
        clusters_max_border=[X[0,:]+self.initial_dist_threshold]
        clusters_min_border=[X[0,:]-self.initial_dist_threshold]
        clusters_submax_border=[X[0,:]+self.initial_dist_threshold*(1+self.incorporate_dist_threshold)]
        clusters_submin_border=[X[0,:]-self.initial_dist_threshold*(1+self.incorporate_dist_threshold)]
        clusters=[1]
        # 遍历每一组向量
        while X.size!=0:
            max_border=clusters_max_border[-1]
            min_border=clusters_min_border[-1]
            id_=np.where(np.sum((X<=max_border)*(X>=min_border),axis=1)!=max_border.size)[0]
            if id_.size==0:
                break
            clusters[-1]+=(X.shape[0]-id_.size)
            X=X[id_,:]
            submax_border=clusters_submax_border[-1]
            submin_border=clusters_submin_border[-1]
            cal=np.sum((X<=submax_border)*(X>=submin_border),axis=1)
            id_=np.where(cal!=submax_border.size)[0]
            if id_.size!=X.shape[0]:
                id_n=np.where(cal!=submax_border.size)[0]
                X_n=X[id_n,:];max_border_n=np.max(X_n,axis=0);min_border_n=np.min(X_n,axis=0)
                clusters_max_border[-1]=max_border+(max_border_n-max_border)*self.expansion_rate
                clusters_min_border[-1]=min_border-(min_border-min_border_n)*self.expansion_rate
                clusters_submax_border[-1]=self.expansion_rate*(clusters_max_border[-1]-clusters_min_border[-1])
                clusters_submin_border[-1]=self.expansion_rate*(clusters_max_border[-1]-clusters_min_border[-1])
                clusters[-1]+=(X.shape[0]-id_.size)
                X=X[id_,:]
            else:
                V=X[0,:];V.reshape(-1);X=X[1:,:]
                clusters_max_border.append(V+self.initial_dist_threshold)
                clusters_min_border.append(V-self.initial_dist_threshold)
                clusters_submax_border.append(V+self.initial_dist_threshold*self.incorporate_dist_threshold)
                clusters_submin_border.append(V-self.initial_dist_threshold*self.incorporate_dist_threshold)
                clusters.append(1)
        self.score_reference=np.append(np.array(clusters_submax_border),np.array(clusters_submin_border),axis=0)
        self.score_reference=np.append(np.zeros([len(clusters_submax_border),clusters_submax_border[0].size]),self.score_reference,axis=0)
        self.score_reference=self.score_reference.reshape(3,len(clusters_submax_border),clusters_submax_border[0].size)
        print('Train over : '+str(len(clusters))+'clusters-'+str(round(time()-t,2))+'s')

    def validate(self,test_X):
        X=(test_X-self.mean)/self.alt;l=self.score_reference.shape[1]
        X=np.repeat(X,3,axis=-1).reshape(X.shape[0],8,3).transpose(0,2,1)
        X=np.repeat(X,l,axis=-1).reshape(X.shape[0],3,8,l).transpose(0,1,3,2)
        score=X-self.score_reference
        score[:,0,:,:]=0;score[:,2,:,:]*=(-1)
        score=np.min(np.linalg.norm(np.max(score,axis=1),axis=2),axis=1)
        # 获取范数最小的分数
        result=np.ones(score.size)
        result[np.where(score<1.15*self.initial_dist_threshold)[0]]=0
        return np.array(result)

    def save(self):
        np.save("results/models/IMS/per_score.npy",self.score_reference)
        np.save("results/models/IMS/per_mean.npy",self.mean)
        np.save("results/models/IMS/per_alt.npy",self.alt)

    def load(self):
        self.score_reference=np.load("results/models/IMS/per_score.npy")
        self.mean=np.load("results/models/IMS/per_mean.npy")
        self.alt=np.load("results/models/IMS/per_alt.npy")

class myIMS:
    def __init__(self):
        self.model = None
        
    def train(self,path_x='X_train.h5',path_y='Y_train.csv'):
        h5_file=h5py.File(path_x,"r");X_train=(np.array(h5_file["data"][:, 2:])).astype('float32')
        Y = pd.read_csv(path_y, header=0, index_col=0).values.astype('float32')
        X_train = X_train.reshape(X_train.shape[0],8,-1,10,10)
        X_train = np.mean(X_train,axis=-1).transpose(0,2,3,1)
        l=Y.shape[1];X_t=np.array([])
        for i in range(l):
            X=X_train[np.where(Y[i,:]==0)[0],i,:,:]
            X=X.reshape(-1,8)
            if X_t.size==0:
                X_t=X.copy()
            else:
                X_t=np.append(X_t,X,axis=0)
        self.model=uniIMS()
        self.model.fit(X_t.reshape(-1,8))
        print('Validation : ')
        X_train=X_train.reshape(-1,8);spli=5*900
        l=int(np.ceil(X_train.shape[0]/spli))
        for i in range(l):
            t=time()
            X=X_train[i*spli:(i+1)*spli,:]
            y = self.model.validate(X)
            y = (np.sum(y.reshape(-1,90,10),axis=2)==0).astype("int32")
            try:
                test_y=np.append(test_y,y,axis=0)
            except:
                test_y=y.copy()
            print("\r\tProcess : "+str(round((i+1)/l*100,2))+"%-"+str(round(time()-t,2))+'s')
        rmse = np.sqrt(mean_squared_error(test_y,Y))
        # Test RMSE = 0.768
        print('Test RMSE: %.3f\n' % rmse)

    def predict(self,path_x='X_train.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        print("Test : ")
        test_X = test_X.reshape(test_X.shape[0],8,-1,10,10)
        test_X = np.mean(test_X,axis=-1).transpose(0,2,3,1).reshape(-1,8)
        spli=5*900;l=int(np.ceil(test_X.shape[0]/spli))
        print(test_X.shape,spli,l,spli*l)
        for i in range(l):
            t=time()
            print(i*spli,(i+1)*spli)
            X=test_X[i*spli:(i+1)*spli,:]
            y = self.model.validate(X)
            y = (np.sum(y.reshape(-1,90,10),axis=2)==0).astype("int32")
            try:
                test_y=np.append(test_y,y,axis=0)
            except:
                test_y=y.copy()
            print("\r\tProcess : "+str(round((i+1)/l*100,2))+"%-"+str(round(time()-t,2))+'s')
        pd.DataFrame(test_y.astype("int32")).to_csv("train/IMS_y.csv")

    def save(self):
        self.model.save()

    def load(self):
        self.model=uniIMS()
        self.model.load()

if __name__ == "__main__":
    ims = myIMS()
    ims.load()
    ims.predict()
