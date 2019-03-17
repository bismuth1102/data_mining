import numpy as np
import pandas as pd
import re as re
import warnings
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)

data = pd.read_csv('data/train_data.csv')
label = pd.read_csv('data/train_label.csv')
tdata = pd.read_csv('data/test_data.csv')
result = pd.read_csv('data/submission.csv')

data.isnull().sum()

#data.loc[data['lvl3'].isnull(),'lvl3']=data[data['lvl3'].isnull()]['lvl2']
data['type'].fillna('international',inplace = True)
tdata['type'].fillna('international',inplace = True)
# data['lvl3'].fillna('unique',inplace = True)
# tdata['lvl3'].fillna('unique',inplace = True)
# # data['lvl2']=data['lvl2'].str.cat(data['lvl1'],sep='_')
# data['lvl3']=data['lvl3'].str.cat(data['lvl2'],sep='_')
# # tdata['lvl2']=tdata['lvl2'].str.cat(tdata['lvl1'],sep='_')
# tdata['lvl3']=tdata['lvl3'].str.cat(tdata['lvl2'],sep='_')
data.head()

tdata.isnull().sum()

XY = pd.merge(data, label, left_index=True, right_index=True, how='outer')
df = XY[['lvl1','lvl2','lvl3','type','price','score']]
df1 = tdata[['lvl1','lvl2','lvl3','type','price']]
XY.head()

X=XY[['lvl1','lvl2','lvl3','price','type']]
y=XY[['score']]
X =pd.get_dummies(X)
tX = pd.get_dummies(df1)
x_columns = [x for x in X.columns]
X = X[x_columns]
y = y['score']
tx_columns = [x for x in tX.columns]
tX = tX[tx_columns]
tX.head()

# 采用DictVectorizer进行特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)

X_train = dict_vec.fit_transform(X.to_dict(orient='record'))
X_test = dict_vec.transform(tX.to_dict(orient='record'))

# 使用随机森林回归模型进行 回归预测
#from sklearn.ensemble import GradientBoostingRegressor
rfr = RandomForestRegressor(n_estimators= 51, max_depth=13, min_samples_split=50,
                                 min_samples_leaf=20,max_features='sqrt',oob_score=True, random_state=10)
#rfr = GradientBoostingRegressor()
rfr.fit(X_train, y)
rfr_y_predict = rfr.predict(X_test)

# 输出结果
Y2 = tdata['id'].values
rfr_submission = pd.DataFrame({'id': Y2, 'Score': rfr_y_predict})
rfr_submission.to_csv('data/submission.csv', index=False)
