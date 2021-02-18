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
        self.mean=np.mean(X_train,axis=0);self.alt=np.max(X_train,axis=0)-np.min(X_train,axis=0)
        self.alt[np.where(self.alt==0)[0]]=0.01
        X=(X_train-self.mean)/self.alt
        # 生成第一个簇
        clusters_max_border=[X[0,:]+self.initial_dist_threshold]
        clusters_min_border=[X[0,:]-self.initial_dist_threshold]
        clusters_submax_border=[X[0,:]+self.initial_dist_threshold*(1+self.incorporate_dist_threshold)]
        clusters_submin_border=[X[0,:]-self.initial_dist_threshold*(1+self.incorporate_dist_threshold)]
        # 遍历每一组向量
        while X.size!=0:
            max_border=clusters_max_border[-1]
            min_border=clusters_min_border[-1]
            id_=np.where(np.sum((X<=max_border)*(X>=min_border),axis=1)!=max_border.size)[0]
            if id_.size==0:
                break
            X=X[id_,:]
            submax_border=clusters_submax_border[-1]
            submin_border=clusters_submin_border[-1]
            cal=np.sum((X<=submax_border)*(X>=submin_border),axis=1)
            print(cal)
            id_=np.where(cal!=submax_border.size)[0]
            if id_.size!=X.shape[0]:
                id_n=np.where(cal!=submax_border.size)[0]
                X_n=X[id_n,:];max_border=np.max(X_n,axis=0);min_border=np.min(X_n,axis=0)
                print(submax_border-max_border,min_border-submin_border)
                clusters_max_border[-1]=max_border+(submax_border-max_border)*self.expansion_rate
                clusters_min_border[-1]=min_border-(min_border-submin_border)*self.expansion_rate
                clusters_submax_border[-1]=self.expansion_rate*(clusters_max_border[-1]-self.clusters_min_border[-1])
                clusters_submin_border[-1]=self.expansion_rate*(clusters_max_border[-1]-self.clusters_min_border[-1])
                X=X[id_,:]
            else:
                V=X[0,:];V.reshape(-1);X=X[1:,:]
                clusters_max_border.append(V+self.initial_dist_threshold)
                clusters_min_border.append(V-self.initial_dist_threshold)
                clusters_submax_border.append(V+self.initial_dist_threshold*self.incorporate_dist_threshold)
                clusters_submin_border.append(V-self.initial_dist_threshold*self.incorporate_dist_threshold) 
        self.score_reference=np.append(np.array(clusters_submax_border),np.array(clusters_submin_border),axis=0)
        self.score_reference=np.append(np.zeros([len(clusters_submax_border),clusters_submax_border[0].size]),self.score_reference,axis=0)
        self.score_reference=self.score_reference.reshape(3,len(clusters_submax_border),clusters_submax_border[0].size)

    def validate(self,test_X):
        X=(test_X-self.mean)/self.alt
        result = [];l=X.shape[0];process=0.1;t=time()
        for i in range(l):
            V = X[i,:]
            compara_matrix=(self.score_reference-V)
            compara_matrix[0,:,:]*=0;compara_matrix[1,:,:]*=-1
            per_cluster_score = np.max(np.max(compara_matrix,axis=0),axis=1)
            # 获取范数最小的分数
            if np.min(per_cluster_score)<1.15*self.initial_dist_threshold:
                result.append(0)
            else:
                result.append(1)
        return np.array(result)

    def save(self,i):
        np.save("results/models/IMS/per_score_"+str(int(i))+".npy",self.score_reference)
        np.save("results/models/IMS/per_mean_"+str(int(i))+".npy",self.mean)
        np.save("results/models/IMS/per_alt_"+str(int(i))+".npy",self.alt)

    def load(self,i):
        self.score_reference=np.load("results/models/IMS/per_score_"+str(int(i))+".npy")
        self.mean=np.load("results/models/IMS/per_mean_"+str(int(i))+".npy")
        self.alt=np.load("results/models/IMS/per_alt_"+str(int(i))+".npy")

class myIMS:
    def __init__(self):
        self.model=[]
        
    def train(self,path_x='X_train.h5',path_y='Y_train.csv'):
        h5_file=h5py.File(path_x,"r");X_train=(np.array(h5_file["data"][:, 2:])).astype('float32')
        Y = pd.read_csv(path_y, header=0, index_col=0).values.astype('float32')
        X_train = X_train.reshape(X_train.shape[0],8,int(X_train.shape[1]/800),100)
        X_train = X_train.transpose(0,2,1,3)
        X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],-1)
        l=Y.shape[1]
        for i in range(l):
            X=X_train[np.where(Y[i,:]==0)[0],i,:]
            self.model.append(uniIMS())
            self.model[-1].fit(X.reshape(X.shape[0],-1))
        print("Validation : ");
        test_y=[];l=X_train.shape[1];
        for i in range(l):
            t=time()
            X=X_train[:,i,:]
            test_y.append(self.model[i].validate(X.reshape(X.shape[0],-1)))
            print("\tColumn "+str(i+1)+"/"+str(l)+" - "+str(round(time()-t,2))+"s")
        rmse = np.sqrt(mean_squared_error(Y, np.array(test_y).astype("int32").T))
        print('Test RMSE: %.3f\n' % rmse)
        self.save()

    def predict(self,path_x='X_test.h5'):
        # load dataset
        h5_file = h5py.File(path_x,"r")
        test_X = (np.array(h5_file["data"][:, 2:])).astype('float32')
        test_X = test_X.reshape(test_X.shape[0],8,int(test_X.shape[1]/800),100)
        test_X = test_X.transpose(0,2,1,3)
        test_X = test_X.reshape(test_X.shape[0],test_X.shape[1],-1)
        test_y = [];l=test_X.shape[1]
        print("Test : ")
        for i in range(l):
            t=time()
            X=test_X[:,i,:]
            test_y.append(self.model[i].validate(X.reshape(X.shape[0],-1)))
            print("\tColumn "+str(i)+"/"+str(l)+" - "+str(round(time()-t,2))+"s")
        pd.DataFrame(np.array(test_y).astype("int32").T).to_csv("results/test_y/IMS_y.csv")

    def save(self):
        for i in range(len(self.model)):
            self.model[i].save(i)

    def load(self):
        self.model=[];i=0
        while True:
            try:
                self.model.append(uniIMS())
                self.model[-1].load(i)
                i+=1
            except:
                break

if __name__ == "__main__":
    ims = myIMS()
    ims.train()
    ims.predict()
