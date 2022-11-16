import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor as GBR
import xgboost
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from catboost import CatBoostRegressor
from math import sqrt
from sklearn import linear_model, tree, ensemble

import warnings
warnings.filterwarnings("ignore")

data=pd.read_csv('C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_clean.csv')
print(data, '\n', data.shape)

x = data.drop(columns = 'BodyFat')
y = data[['BodyFat']]
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3,random_state=2022)
print('data shape after split:', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# Save the test set data and training set data for later parameter adjustment and evaluation
data_train=pd.concat([x_train, y_train], axis=1)
data_test=pd.concat([x_test, y_test], axis=1)
print(data_train.shape, data_test.shape)
data_train.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_train.csv', index=None)
data_test.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_test.csv', index=None)


kf =KFold(n_splits=5, shuffle=True, random_state=2022)
cnt=1
for train_index, test_index in kf.split(x_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1
def rmse(score):
    rmse = np.sqrt(-score)
    print(f'rmse= {"{:.2f}".format(rmse)}')

# training model and select best model
## 多元线性回归
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
score_data1=pd.DataFrame()
for sco in scoring:
    score = cross_val_score(linear_model.LinearRegression(), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco == "neg_mean_squared_error":
        score = math.sqrt(-(score.mean()))
    if sco == "neg_mean_absolute_error":
        score = -score.mean()
    if sco == "r2":
        score = score.mean()
    # print(sco, 'of LR : ', score)
    score_data1 = score_data1.append(pd.DataFrame({'LR': [score]}), ignore_index=True)
# print(score_data1)

### K-近邻回归
score_data2=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(KNeighborsRegressor(), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco == "neg_mean_squared_error":
        score = math.sqrt(-(score.mean()))
    if sco == "neg_mean_absolute_error":
        score = -score.mean()
    if sco == "r2":
        score = score.mean()
    # print(sco, 'of KNN : ', score)
    score_data2 = score_data2.append(pd.DataFrame({'KNN': [score]}), ignore_index=True)

### 神经网络
score_data3=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(MLPRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco == "neg_mean_squared_error":
        score = math.sqrt(-(score.mean()))
    if sco == "neg_mean_absolute_error":
        score = -score.mean()
    if sco == "r2":
        score = score.mean()
    # print(sco, 'of MLP : ', score)
    score_data3 = score_data3.append(pd.DataFrame({'MLP': [score]}), ignore_index=True)

### DT
score_data4=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(DecisionTreeRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco == "neg_mean_squared_error":
        score = math.sqrt(-(score.mean()))
    if sco == "neg_mean_absolute_error":
        score = -score.mean()
    if sco == "r2":
        score = score.mean()
    # print(sco, 'of DT : ', score)
    score_data4 = score_data4.append(pd.DataFrame({'DT': [score]}), ignore_index=True)

### Adaboost
score_data5=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(AdaBoostRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco == "neg_mean_squared_error":
        score = math.sqrt(-(score.mean()))
    if sco == "neg_mean_absolute_error":
        score = -score.mean()
    if sco == "r2":
        score = score.mean()
    # print(sco, 'of Ada : ', score)
    score_data5 = score_data5.append(pd.DataFrame({'Adaboost': [score]}), ignore_index=True)

### GBR
score_data6=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(GBR(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco=="neg_mean_squared_error":
        score=math.sqrt(-(score.mean()))
    if sco=="neg_mean_absolute_error":
        score=-score.mean()
    if sco=="r2":
        score=score.mean()
    # print(sco, 'of GBR : ', score)
    score_data6 = score_data6.append(pd.DataFrame({'GBR': [score]}), ignore_index=True)

### XGboost
score_data7=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(xgboost.XGBRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco=="neg_mean_squared_error":
        score=math.sqrt(-(score.mean()))
    if sco=="neg_mean_absolute_error":
        score=-score.mean()
    if sco=="r2":
        score=score.mean()
    # print(sco, 'of XGboost : ', score)
    score_data7 = score_data7.append(pd.DataFrame({'XGboost': [score]}), ignore_index=True)

### RF
score_data8=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(RandomForestRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco=="neg_mean_squared_error":
        score=math.sqrt(-(score.mean()))
    if sco=="neg_mean_absolute_error":
        score=-score.mean()
    if sco=="r2":
        score=score.mean()
    # print(sco, 'of RF : ', score)
    score_data8 = score_data8.append(pd.DataFrame({'RF': [score]}), ignore_index=True)

### LGBMR
score_data9=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(LGBMRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco=="neg_mean_squared_error":
        score=math.sqrt(-(score.mean()))
    if sco=="neg_mean_absolute_error":
        score=-score.mean()
    if sco=="r2":
        score=score.mean()
    # print(sco, 'of LGBMR : ', score)
    score_data9 = score_data9.append(pd.DataFrame({'LGBM': [score]}), ignore_index=True)

### SVR
score_data10=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(SVR(), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco=="neg_mean_squared_error":
        score=math.sqrt(-(score.mean()))
    if sco=="neg_mean_absolute_error":
        score=-score.mean()
    if sco=="r2":
        score=score.mean()
    # print(sco, 'of SVR : ', score)
    score_data10 = score_data10.append(pd.DataFrame({'SVR': [score]}), ignore_index=True)

## ExtraTreesRegressor
score_data11=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(ExtraTreesRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco=="neg_mean_squared_error":
        score=math.sqrt(-(score.mean()))
    if sco=="neg_mean_absolute_error":
        score=-score.mean()
    if sco=="r2":
        score=score.mean()
    # print(sco, 'of ETR : ', score)
    score_data11 = score_data11.append(pd.DataFrame({'ETR': [score]}), ignore_index=True)


### CatBoost
score_data12=pd.DataFrame()
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
for sco in scoring:
    score = cross_val_score(CatBoostRegressor(random_state=2022), x_train, y_train, cv=kf, scoring=sco, verbose=False)
    #print(f'Scores for each fold: {score}')
    if sco=="neg_mean_squared_error":
        score=math.sqrt(-(score.mean()))
    if sco=="neg_mean_absolute_error":
        score=-score.mean()
    if sco=="r2":
        score=score.mean()
    # print(sco, 'of CatBoost : ', score)
    score_data12 = score_data12.append(pd.DataFrame({'Catboost': [score]}), ignore_index=True)

# The evaluation indexes of all models were summarized
score_data=pd.concat([score_data1, score_data2, score_data3, score_data4,
                     score_data5, score_data6, score_data7, score_data8, score_data9,
                      score_data10, score_data11, score_data12, ], axis=1)
score_Data=score_data.rename(index={0:'MSE', 1:'MAE', 2:'r2'}).T
print(score_Data)

score_Data.to_csv(path_or_buf=r'C:\Users\Gealen\PycharmProjects\pythonProject\BodyFat\data\data_score.csv', index=True)