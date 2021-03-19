import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def chooseModel(X_train,y_train):
    # # 看看默认设置效果
    # rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    # rf0.fit(X_train,y_train)
    # print ("rf0:",rf0.oob_score_)

    # rf1 = RandomForestClassifier(oob_score=True,n_estimators=300,random_state=10,
    #                              max_features='sqrt',n_jobs=-1,verbose=1)
    # rf1.fit(X_train,y_train)
    # print ("rf1:",rf1.oob_score_)


    # # 对n_estimators调整
    # param_test1 = {'n_estimators':range(50,350,50)}
    # gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
    #                                       min_samples_leaf=20,max_depth=8,max_features='sqrt',
    #                                       random_state=10),
    #                                           param_grid = param_test1, scoring='roc_auc',cv=5)
    # gsearch1.fit(X_train,y_train)
    # print("grid_scores:       best parameters:       best score: ")
    # print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)

    # 在41附近缩小n_estimators的范围为30-49
    score_lt = []
    for i in range(260,280):
        rfc = RandomForestClassifier(n_estimators=i
                                     ,random_state=90)
        score = cross_val_score(rfc, X_train, y_train, cv=10).mean()
        score_lt.append(score)
    score_max = max(score_lt)
    print('max score：{}'.format(score_max),
          'used trees：{}'.format(score_lt.index(score_max)+260))

    # 绘制学习曲线
    x = np.arange(260,280)
    plt.subplot(111)
    plt.plot(x, score_lt,'o-')
    plt.show()

def supertrain(X_data,y_data,X_test,split,suffix="submission"):     # 加个suffix提醒自己改名。。。
    if(split==1):
        model = RandomForestClassifier(n_estimators=271,random_state=10,
                                       max_features='sqrt',n_jobs=-1,verbose=1)
        X_train, myX_test, y_train, myy_test = train_test_split(X_data, y_data, test_size=0.3)
        pd.DataFrame(myy_test).to_csv("myanswer.csv",index=True,sep=',')#保存myy_test作为标准答案便于比较
        model.fit(X_train,y_train)# 训练模型
        path_modle = 'MymodelsTemp/myrf.pkl' # 保存模型
        joblib.dump(model,path_modle)
        predict(path_modle,myX_test,"myy_test.csv") # 预测
    else:
        model = RandomForestClassifier(n_estimators=300,random_state=None,
                                       max_features='sqrt',n_jobs=-1,verbose=1)
        X_train = X_data
        y_train = y_data
        model.fit(X_train,y_train)
        path_modle = 'Mymodels/myrf.pkl'
        joblib.dump(model,path_modle)
        path_out = suffix+".csv"
        predict(path_modle,X_test,path_out)
        transform_file(path_in,path_out)


def predict(modelfile,X_test,path_ytest):
    y_evaluate = joblib.load(modelfile).predict(X_test)
    pd.DataFrame(y_evaluate).to_csv(path_ytest,index=True,sep=',')

def transform_file(pathin,pathout):
    Y_benchmark=pd.read_csv(pathin, header=0, index_col=0)
    Y = pd.DataFrame(pd.read_csv(pathout,
                                 header=0, index_col=0).values.astype('int32'),
                     columns=Y_benchmark.columns)
    Y.rename(index={i:i+4400 for i in Y.index},inplace=True) # index即第一列改为和benchmark一样，从4400开始
    Y.to_csv(pathout)

##########################################请开始你的表演#########################################
path_x='E:\S8\ML\Dreem\X_train.h5'
path_y='E:\S8\ML\Dreem\y_train.csv'
path_in = "E:\S8\ML\Dreem\y_benchmark.csv"

# 用h5_file.keys()看到只有一个data set 即 'data',大小为4400x72002,将其转为array类型
h5_file=h5py.File(path_x,"r")
X_data=(np.array(h5_file['data'][:, 2:])).astype('float32') # 从第二列开始取，4400x72000
y_data = pd.read_csv(path_y, header=0, index_col=0).values.astype('int32') # 4400x90
h5_file=h5py.File('E:/S8/ML/Dreem/X_test.h5',"r")
X_test=(np.array(h5_file['data'][:, 2:])).astype('float32')
x = np.vstack((X_data,X_test))
x = PCA(n_components=12).fit_transform(x)
x = TSNE(n_components=2).fit_transform(x)
X_data = x[:4400,:]
X_test = x[4400:,:]

# print(pca.explained_variance_ratio_)
#
# chooseModel(X_data,y_data)
# supertrain(X_data,y_data,1,pca,tsne)
supertrain(X_data,y_data,X_test,0,"submission8")
