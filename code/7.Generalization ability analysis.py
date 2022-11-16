import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
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
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy.stats as stata
import seaborn as sns

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# Read data
data_train = pd.read_csv('data/data_train.csv')
data_test = pd.read_csv('data/data_test.csv')

x_train=data_train.drop('BodyFat', axis=1)
y_train=data_train['BodyFat']

x_test=data_test.drop('BodyFat', axis=1)
y_test=data_test['BodyFat']

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Models
# Linear regression
LR=linear_model.LinearRegression()
LR.fit(x_train, y_train)
y_train_pred_LR=LR.predict(x_train)
y_test_pred_LR=LR.predict(x_test)
MSE_LR=mean_squared_error(y_test, y_test_pred_LR, squared=False)
MAE_LR=mean_absolute_error(y_test, y_test_pred_LR)
R2_LR=r2_score(y_test, y_test_pred_LR)
print(MSE_LR, MAE_LR, R2_LR)

# KNN
KNN=KNeighborsRegressor()

# SVR
SVR=SVR()
SVR.fit(x_train, y_train)
y_train_pred_SVR=SVR.predict(x_train)
y_test_pred_SVR=SVR.predict(x_test)
MSE_SVR=mean_squared_error(y_test, y_test_pred_SVR, squared=False)
MAE_SVR=mean_absolute_error(y_test, y_test_pred_SVR)
R2_SVR=r2_score(y_test, y_test_pred_SVR)
print(MSE_SVR, MAE_SVR, R2_SVR)
# MLP
MLP=MLPRegressor(random_state=2022)

# DT
DT=DecisionTreeRegressor(random_state=2022)

# XG
XG=xgboost.XGBRegressor(random_state=2022)
XG.fit(x_train, y_train)
y_train_pred_XG=XG.predict(x_train)
y_test_pred_XG=XG.predict(x_test)
MSE_XG=mean_squared_error(y_test, y_test_pred_XG, squared=False)
MAE_XG=mean_absolute_error(y_test, y_test_pred_XG)
R2_XG=r2_score(y_test, y_test_pred_XG)
print(MSE_XG, MAE_XG, R2_XG)

# LGBM
LGBM=LGBMRegressor(random_state=2022)
LGBM.fit(x_train, y_train)
y_train_pred_LGBM=LGBM.predict(x_train)
y_test_pred_LGBM=LGBM.predict(x_test)
MSE_LGBM=mean_squared_error(y_test, y_test_pred_LGBM, squared=False)
MAE_LGBM=mean_absolute_error(y_test, y_test_pred_LGBM)
R2_LGBM=r2_score(y_test, y_test_pred_LGBM)
print(MSE_LGBM, MAE_LGBM, R2_LGBM)

# Adaboost
Ada=AdaBoostRegressor(random_state=2022)

# Catboost
Cat=CatBoostRegressor(random_state=2022)

# GBR
GBR=GBR(random_state=2022)

# RF
RF=RandomForestRegressor(random_state=2022)

# ETR
ETR=ExtraTreesRegressor(n_estimators=130,
                        max_depth=16,
                        max_features=15,
                        min_samples_leaf=1,
                        min_samples_split=2,
                        random_state=2022
                        )
ETR.fit(x_train, y_train)
y_train_pred_ETR=ETR.predict(x_train)
y_test_pred_ETR=ETR.predict(x_test)
MSE_ETR=mean_squared_error(y_test, y_test_pred_ETR, squared=False)
MAE_ETR=mean_absolute_error(y_test, y_test_pred_ETR)
R2_ETR=r2_score(y_test, y_test_pred_ETR)
print(MSE_ETR, MAE_ETR, R2_ETR)

# Ture and predicted values plot
def Ture_predict_plot(y_train_true, y_train_pred, y_test_true, y_test_pred, title, *args, **kwargs):
    fig = plt.figure(figsize=(8, 6))
    ax1 = plt.scatter(y_train_true, y_train_pred, marker='X', color="blue", )
    ax2 = plt.scatter(y_test_true, y_test_pred, s=100, marker='*', color="darkorange", alpha=1, label="Test set")
    plt.plot(y_train_true, y_train_true, color='red')
    plt.xlabel('True values', fontsize=15)
    plt.ylabel('Predicted values', fontsize=15)
    plt.title(title, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend((ax1, ax2), ('Training set', 'Test set'), loc='best', fontsize=15)
    plt.show()


print('The Correlation of y_test and y_pred：', stata.pearsonr(y_test, y_test_pred_ETR))

# Bland-Altman plot
def bland(data1, data2, *args, **kwargs):
    # data1=np.asarray(data1.reshape(-1))
    # data2=np.asarray(data2.reshape(-1))
    mean=np.mean([data1, data2], axis=0)
    diff=data1-data2

    md=np.mean(diff)
    print(md)
    sd=np.std(diff, axis=0)
    plt.figure(figsize=(12, 8))

    plt.scatter(mean, diff, marker='o', s=120, c='b', edgecolors='r', *args, **kwargs)

    plt.axhline(md, color='gray', linestyle='--', label='mean(diff)')
    plt.text(mean.max() - 4.5, md + 0.2, 'mean(diff)=0.08', fontsize=15)

    plt.axhline(0, color='g', linestyle='-', label='mean')
    plt.text(mean.max() - 4.5, md - 0.5, 'mean=0', fontsize=15)

    plt.axhline(md+1.96*sd, color='red', label='mean(diff)+1.96*sd')
    plt.text(mean.max()-8, (md + 1.96 * sd) + 0.2, 'mean(diff)+1.96*sd', fontsize=15)

    plt.axhline(md-1.96*sd, color='red', label='mean(diff)-1.96*sd')
    plt.text(mean.max()-8, (md - 1.96 * sd) + 0.2, 'mean(diff)-1.96*sd', fontsize=15)

    plt.xlabel('Mean of true and predicted values', fontsize=15)
    plt.ylabel('diff of true and predicted values', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Bland-Altman plot', fontsize=15)
    plt.show()

Ture_predict_plot(y_train, y_train_pred_ETR, y_test, y_test_pred_ETR, title='ETR')
Ture_predict_plot(y_train, y_train_pred_LR, y_test, y_test_pred_LR, title='LR')
Ture_predict_plot(y_train, y_train_pred_XG, y_test, y_test_pred_XG, title='XG')
Ture_predict_plot(y_train, y_train_pred_LGBM, y_test, y_test_pred_LGBM, title='LGBM')

bland(y_test, y_test_pred_ETR)