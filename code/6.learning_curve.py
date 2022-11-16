import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
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
from catboost import CatBoostRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import scikitplot as skplt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

data_train=pd.read_csv('data/data_train.csv')
x_train=data_train.drop('BodyFat', axis=1)
y_train=data_train['BodyFat']
print(x_train.shape, y_train.shape)

# Models
# Linear regression
LR=linear_model.LinearRegression()

# KNN
KNN=KNeighborsRegressor()

# SVR
SVR=SVR()

# MLP
MLP=MLPRegressor(random_state=2022)

# DT
DT=DecisionTreeRegressor(random_state=2022)

# XG
XG=xgboost.XGBRegressor(random_state=2022)

# LGBM
LGBM=LGBMRegressor(random_state=2022)

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

# K-fold
kf =KFold(n_splits=5, shuffle=True, random_state=2022)
cnt=1
for train_index, test_index in kf.split(x_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

#
skplt.estimators.plot_learning_curve(LR, x_train, y_train, cv=kf, scoring='r2', random_state=2022, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(a) LR', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(KNN, x_train, y_train, cv=kf, scoring='r2', random_state=2022, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(b) KNN', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(SVR, x_train, y_train, cv=kf, scoring='r2', random_state=2022, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(c) SVR', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(XG, x_train, y_train, cv=kf, scoring='r2', random_state=2022, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(d) XGboost', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(LGBM, x_train, y_train, cv=kf, scoring='r2', random_state=2022, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(e) LightGBM', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()

#
skplt.estimators.plot_learning_curve(RF, x_train, y_train, cv=kf, scoring='r2', random_state=2022, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(e1) RF', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()
#
skplt.estimators.plot_learning_curve(ETR, x_train, y_train, cv=kf, scoring='r2', random_state=2022, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10))
plt.legend(loc='lower right', fontsize=15)
plt.xlabel('Training examples', fontsize=15)
plt.ylabel('Score', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(f) IBFPPM', y=-0.2, fontsize=15)
plt.tight_layout()
plt.show()