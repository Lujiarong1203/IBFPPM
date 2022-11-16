import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import validation_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
import math
plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# Load the data
data = pd.read_csv('C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_train.csv')
print(data.shape)

y_train=data['BodyFat']
x_train=data.drop('BodyFat', axis=1)
print(x_train.shape, y_train.shape)

# K-fold
kf =KFold(n_splits=5, shuffle=True, random_state=2022)
cnt=1
for train_index, test_index in kf.split(x_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# Parameter tuning/Each time a parameter is tuned, update the parameter corresponding to other_params to the optimal value
# Use more trees (n_estimators)
cv_params= {'n_estimators': [100, 110, 120, 130, 140, 150, 160, 170]}

model = ExtraTreesRegressor(random_state=2022)
optimized_ETR = GridSearchCV(estimator=model, param_grid=cv_params, scoring="r2", cv=kf, verbose=1, n_jobs=-1)
optimized_ETR.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_ETR.best_params_))
print('Best model score:{0}'.format(optimized_ETR.best_score_))
# The best value of the parameter：{'n_estimators': 130}
# Best model score:0.9733234189697937

# Draw the n_estimators validation_curve
param_range_1=[100, 110, 120, 130, 140, 150, 160, 170]
train_scores_1, test_scores_1 = validation_curve(estimator=model,
                                             X=x_train,
                                             y=y_train,
                                             param_name='n_estimators',
                                             param_range=param_range_1,
                                             cv=kf, scoring='r2', n_jobs=-1)

# print(train_scores_1, test_scores_1)

train_mean_1=np.mean(train_scores_1, axis=1)
train_std_1=np.std(train_scores_1, axis=1)
test_mean_1=np.mean(test_scores_1, axis=1)
test_std_1=np.std(test_scores_1, axis=1)

plt.plot(param_range_1, train_mean_1, color="darkorange", linewidth=3.0,
         marker='X', markersize=10, label='training score')

plt.fill_between(param_range_1, train_mean_1 + train_std_1,
                 train_mean_1 - train_std_1, alpha=0.1, color="darkorange")

plt.plot(param_range_1, test_mean_1, color="blue", linewidth=3.0,
         marker='d', markersize=10,label='validation score')

plt.fill_between(param_range_1,test_mean_1 + test_std_1,
                 test_mean_1 - test_std_1, alpha=0.1, color="blue")

plt.grid(b=True, axis='y')
# plt.xscale('log')
plt.legend(loc='best', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('R^2', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(a) n_estimators', y=-0.2, fontsize=15)
plt.ylim([0.97, 1.0])
plt.tight_layout()
plt.show()
#
#
# tuning max_depth
cv_params= {'max_depth': [10, 11, 12, 13, 14, 15, 16, 17, 18]}

model = ExtraTreesRegressor(n_estimators=130, random_state=2022)
optimized_ETR = GridSearchCV(estimator=model, param_grid=cv_params, scoring="r2", cv=kf, verbose=1, n_jobs=-1)
optimized_ETR.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_ETR.best_params_))   # The best value of the parameter：{'max_depth': 18}
print('Best model score:{0}'.format(optimized_ETR.best_score_))                   # Best model score:0.9751704494744902

# Draw the max_depth validation curve
param_range_2=[10, 11, 12, 13, 14, 15, 16, 17, 18]
train_scores_2, test_scores_2 = validation_curve(estimator=model,
                                             X=x_train,
                                             y=y_train,
                                             param_name='max_depth',
                                             param_range=param_range_2,
                                             cv=kf, scoring='r2', n_jobs=-1)

train_mean_2=np.mean(train_scores_2, axis=1)
train_std_2=np.std(train_scores_2, axis=1)
test_mean_2=np.mean(test_scores_2, axis=1)
test_std_2=np.std(test_scores_2, axis=1)

plt.plot(param_range_2, train_mean_2, color="darkorange", linewidth=3.0,
         marker='X', markersize=10, label='training score')

plt.fill_between(param_range_2, train_mean_2 + train_std_2,
                 train_mean_2 - train_std_2, alpha=0.1, color="darkorange")

plt.plot(param_range_2, test_mean_2, color="blue", linewidth=3.0,
         marker='d', markersize=10, label='validation score')

plt.fill_between(param_range_2, test_mean_2 + test_std_2,
                 test_mean_2 - test_std_2, alpha=0.1, color="blue")

plt.grid(b=True, axis='y')
# plt.xscale('log')
plt.legend(loc='best', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('R^2', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(b) max_depth', y=-0.2, fontsize=15)
plt.ylim([0.965, 1.0])
plt.tight_layout()
plt.show()


# tuning max_features
cv_params= {'max_features': [8, 9, 10, 11, 12, 13, 14, 15]}

model = ExtraTreesRegressor(n_estimators=130, max_depth=16, random_state=2022)
optimized_ETR = GridSearchCV(estimator=model,
                            param_grid=cv_params,
                            scoring="r2",
                            cv=kf,
                            verbose=1, n_jobs=-1)
optimized_ETR.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_ETR.best_params_))
print('Best model score:{0}'.format(optimized_ETR.best_score_))
#
# Draw the max_features validation curve
param_range_3=[8, 9, 10, 11, 12, 13, 14, 15]
train_scores_3, test_scores_3 = validation_curve(estimator=model,
                                             X=x_train,
                                             y=y_train,
                                             param_name='max_features',
                                             param_range=param_range_3,
                                             cv=kf, scoring='r2', n_jobs=-1)

train_mean_3=np.mean(train_scores_3, axis=1)
train_std_3=np.std(train_scores_3, axis=1)
test_mean_3=np.mean(test_scores_3, axis=1)
test_std_3=np.std(test_scores_3, axis=1)

plt.plot(param_range_3, train_mean_3, color="darkorange", linewidth=3.0,
         marker='X', markersize=10, label='training score')

plt.fill_between(param_range_3, train_mean_3 + train_std_3,
                 train_mean_3 - train_std_3, alpha=0.1, color="darkorange")

plt.plot(param_range_3, test_mean_3, color="blue", linewidth=3.0,
         marker='d', markersize=10, label='validation score')

plt.fill_between(param_range_3, test_mean_3 + test_std_3,
                 test_mean_3 - test_std_3, alpha=0.1, color="blue")

plt.grid(b=True, axis='y')
# plt.xscale('log')
plt.legend(loc='best', fontsize=15)
plt.xlabel('parameter values', fontsize=15)
plt.ylabel('R^2', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('(c) max_features', y=-0.2, fontsize=15)
plt.ylim([0.95, 1.0])
plt.tight_layout()
plt.show()

# tuning min_samples_leaf
cv_params= {'min_samples_leaf': [1, 2, 3, 4]}
model = ExtraTreesRegressor(n_estimators=130, max_depth=16, max_features=15, random_state=2022)
optimized_ETR = GridSearchCV(estimator=model,
                            param_grid=cv_params,
                            scoring="r2",
                            cv=kf,
                            verbose=1, n_jobs=-1)
optimized_ETR.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_ETR.best_params_))
print('Best model score:{0}'.format(optimized_ETR.best_score_))

# 调试min_samples_split
cv_params= {'min_samples_split': [2, 3, 4, 5, 6]}

model = ExtraTreesRegressor(n_estimators=130, max_depth=16, max_features=15, min_samples_leaf=1, random_state=2022)
optimized_ETR = GridSearchCV(estimator=model,
                            param_grid=cv_params,
                            scoring="r2",
                            cv=kf,
                            verbose=1, n_jobs=-1)
optimized_ETR.fit(x_train, y_train)
print('The best value of the parameter：{0}'.format(optimized_ETR.best_params_))
print('Best model score:{0}'.format(optimized_ETR.best_score_))

# best parameters: (n_estimators=130, max_depth=16, max_features=15, min_samples_leaf=1, min_samples_split=2, random_state=2022)

# Verify that the optimal parameters improve the effect
data_test=pd.read_csv('data/data_test.csv')
x_test=data_test.drop(['BodyFat'], axis=1)
y_test=data_test['BodyFat']
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# compara the result of Hyperparameter
# Before Tuning
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
score_data1=pd.DataFrame()
for sco in scoring:
    score = cross_val_score(ExtraTreesRegressor(random_state=2022),
                            x_train, y_train, cv=kf, scoring=sco)
    #print(f'Scores for each fold: {score}')
    if sco == "neg_mean_squared_error":
        score = math.sqrt(-(score.mean()))
    if sco == "neg_mean_absolute_error":
        score = -score.mean()
    if sco == "r2":
        score = score.mean()
    score_data1 = score_data1.append(pd.DataFrame({'Before tuning': [score]}), ignore_index=True)
print(score_data1)

# After Tuning
scoring=["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
score_data2=pd.DataFrame()
for sco in scoring:
    score = cross_val_score(ExtraTreesRegressor(n_estimators=130,
                                                max_depth=16,
                                                max_features=15,
                                                min_samples_leaf=1,
                                                min_samples_split=2,
                                                random_state=2022
                                                ),
                            x_train, y_train, cv=kf, scoring=sco
                            )
    #print(f'Scores for each fold: {score}')
    if sco == "neg_mean_squared_error":
        score = math.sqrt(-(score.mean()))
    if sco == "neg_mean_absolute_error":
        score = -score.mean()
    if sco == "r2":
        score = score.mean()
    score_data2 = score_data2.append(pd.DataFrame({'After tuning': [score]}), ignore_index=True)

score_com=pd.concat([score_data1, score_data2], axis=1)
score_COM=score_com.rename(index={0:'MSE', 1:'MAE', 2:'R2'})
print(score_COM.T)