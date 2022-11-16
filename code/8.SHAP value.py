import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import xgboost
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from math import sqrt
import shap

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

# ETR
ETR=ExtraTreesRegressor(n_estimators=110,
                        max_depth=11,
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

# SHAP
# SHAP summary plot
explainer = shap.TreeExplainer(ETR)
shap_values = explainer.shap_values(x_train)
print(shap_values.shape)
shap.summary_plot(shap_values, x_train)

# SHAP dependency plot
shap.dependence_plot('Density', shap_values, x_train, interaction_index='Abdomen')
shap.dependence_plot('Abdomen', shap_values, x_train, interaction_index='BMI')
shap.dependence_plot('BMI', shap_values, x_train, interaction_index='Density')


shap.force_plot(explainer.expected_value[0],
                shap_values[3,:],
                x_train.iloc[3,:],
                text_rotation=20,
                matplotlib=True)

shap.force_plot(explainer.expected_value[0],
                shap_values[7,:],
                x_train.iloc[7,:],
                text_rotation=20,
                matplotlib=True)
