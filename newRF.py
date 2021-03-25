import h5py
import metric_dreem as mtr
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def chooseModel(X_train,y_train):
    # rf0 = RandomForestClassifier(oob_score=True, random_state=10)
    # rf0.fit(X_train,y_train)
    # print (rf0.oob_score_)
    # # y_predprob = rf0.predict_proba(X_train)[:,1]
    # # print ("AUC Score (Train): %f" % metrics.roc_auc_score(y_train, y_predprob))
    param_test1 = {'n_estimators':range(100,320,20)}
    gsearch1 = GridSearchCV(estimator = RandomForestClassifier(max_features='sqrt' ,random_state=10),
                            param_grid = param_test1, scoring='roc_auc',cv=5)
    gsearch1.fit(X_train,y_train)
    print("gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_")
    print(gsearch1.scoring, gsearch1.best_params_, gsearch1.best_score_)

def choose_pca():
    # 用h5_file.keys()看到只有一个data set 即 'data',大小为4400x72002,将其转为array类型
    h5_file=h5py.File(path_x,"r")
    X_data=(np.array(h5_file['data'][:, 2:])).astype('float32') # 从第二列开始取，4400x72000
    y_data = pd.read_csv(path_y, header=0, index_col=0).values.astype('int32') # 4400x90
    pca = PCA()
    # pca = PCA(n_components=80)
    pca.fit(X_data,y_data)
    ratio=pca.explained_variance_ratio_
    print("pca.components_",pca.components_.shape)
    print("pca_var_ratio",pca.explained_variance_ratio_.shape)
    #绘制图形
    plt.plot([i for i in range(X_data.shape[1])],
             [np.sum(ratio[:i+1]) for i in range(X_data.shape[1])])
    plt.xticks(np.arange(X_data.shape[1]))
    plt.axes(xscale = "log")
    plt.yticks(np.arange(0,1.01,0.05))
    plt.xlabel("n_components")
    plt.ylabel("explained variance ratio")
    plt.grid()
    plt.show()

def choose_train(X_data,y_data):
    X_train, myX_test, y_train, myy_test = train_test_split(X_data, y_data, test_size=0.2,random_state=40)
    dreem_score = []
    model = RandomForestClassifier(n_estimators=300,n_jobs=-1,verbose=1,random_state=40,max_features='sqrt')
    for i in range(10,31,2):
        # model = RandomForestClassifier(n_estimators=300,random_state=90,
        #                                max_features='sqrt',n_jobs=-1,verbose=1)
        pca = PCA(n_components=i)
        X_data= pca.fit_transform(X_data)
        X_data = TSNE(n_components=2).fit_transform(X_data)
        pd.DataFrame(myy_test).to_csv("myanswer.csv",index=True,sep=',')#保存myy_test作为标准答案便于比较
        model.fit(X_train,y_train)# 训练模型
        y_evaluate = model.predict(myX_test)
        pd.DataFrame(y_evaluate).to_csv("myy_test.csv",index=True,sep=',')
        df_y_true = pd.read_csv('myanswer.csv', index_col=0, sep=',')
        df_y_pred = pd.read_csv('myy_test.csv', index_col=0, sep=',')
        score = mtr.dreem_sleep_apnea_custom_metric(df_y_true, df_y_pred)
        print("score:",score)
        dreem_score = np.append(dreem_score,score)

    print("best score:",np.max(dreem_score))
    plt.plot([i for i in range(14,30,2)],
             dreem_score)
    plt.xlabel("n_components")
    plt.ylabel("dreem_score")
    plt.grid()
    plt.show()

def train(X_data,y_data):
    X_train, myX_test, y_train, myy_test = train_test_split(X_data, y_data, test_size=0.2,random_state=40)
    model = RandomForestClassifier(n_estimators=300,n_jobs=-1,verbose=1,random_state=40,max_features='sqrt')
    pd.DataFrame(myy_test).to_csv("myanswer.csv",index=True,sep=',')#保存myy_test作为标准答案便于比较
    model.fit(X_train,y_train)# 训练模型
    y_evaluate = model.predict(myX_test)
    pd.DataFrame(y_evaluate).to_csv("myy_test.csv",index=True,sep=',')
    df_y_true = pd.read_csv('myanswer.csv', index_col=0, sep=',')
    df_y_pred = pd.read_csv('myy_test.csv', index_col=0, sep=',')
    score = mtr.dreem_sleep_apnea_custom_metric(df_y_true, df_y_pred)
    print("score:",score)

def supertrain(suffix):
    # 用h5_file.keys()看到只有一个data set 即 'data',大小为4400x72002,将其转为array类型
    h5_file=h5py.File(path_x,"r")
    X_data=(np.array(h5_file['data'][:, 2:])).astype('float32') # 从第二列开始取，4400x72000
    y_data = pd.read_csv(path_y, header=0, index_col=0).values.astype('int32') # 4400x90
    h5_file=h5py.File('E:/S8/ML/Dreem/X_test.h5',"r")
    X_test=(np.array(h5_file['data'][:, 2:])).astype('float32')
    x = np.vstack((X_data,X_test))
    x = PCA(n_components=20).fit_transform(x)
    x = TSNE(n_components=2).fit_transform(x)
    X_data = x[:4400,:]
    X_test = x[4400:,:]
    # model = RandomForestClassifier(n_estimators=300,random_state=None,
    #                                max_features='sqrt',n_jobs=-1,verbose=1)
    model = RandomForestClassifier(n_estimators=320,n_jobs=-1,verbose=1)
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

# choose_pca()
# chooseModel(X_data,y_data)

# 用h5_file.keys()看到只有一个data set 即 'data',大小为4400x72002,将其转为array类型
# h5_file=h5py.File(path_x,"r")
# X_data=(np.array(h5_file['data'][:, 2:])).astype('float32') # 从第二列开始取，4400x72000
# y_data = pd.read_csv(path_y, header=0, index_col=0).values.astype('int32') # 4400x90
# pca = PCA(n_components=0.95)
# print("pca.components:",pca.components_.shape)

# choose_train(X_data,y_data)

h5_file=h5py.File(path_x,"r")
X_data=(np.array(h5_file['data'][:, 2:])).astype('float32') # 从第二列开始取，4400x72000
y_data = pd.read_csv(path_y, header=0, index_col=0).values.astype('int32') # 4400x90
pca = PCA(n_components=20)
X_data = TSNE(n_components=2).fit_transform(X_data)
train(X_data,y_data)

# supertrain("submission9")
