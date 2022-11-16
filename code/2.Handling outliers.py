import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data=pd.read_csv('data/data_1.csv')
pd.set_option('display.max_columns', None)

## 查看异常值
list1=['Weight', 'Height', 'Abdomen', 'Thigh']
print(data[list1])
fig,axes = plt.subplots(len(list1),1,figsize=(16,10))
plt.subplots_adjust(hspace=1)
for i, axe in enumerate(list1):
    sns.boxplot(data[axe],
                whis=2,
                orient='h',
                palette='OrRd',
                fliersize=10,
                ax=axes[i],
                )
    axes[i].tick_params(axis='x', labelsize=15)
    axes[i].set_xlabel(xlabel=axe, fontsize=15)
plt.show()
del list1

list2=['Chest', 'Hip', 'Knee', 'Ankle']
fig,axes = plt.subplots(len(list2),1,figsize=(16,10))
plt.subplots_adjust(hspace=1)
for i, axe in enumerate(list2):
    sns.boxplot(data[axe],
                whis=2,
                orient='h',
                palette='OrRd',
                fliersize=10,
                ax=axes[i],
                )
    axes[i].tick_params(axis='x', labelsize=15)
    axes[i].set_xlabel(xlabel=axe, fontsize=15)
plt.show()

IQ=data.describe().T[['25%', '75%']]
IQ=pd.DataFrame(IQ)
low=IQ['25%'] - 1.5 * (IQ['75%'] - IQ['25%'])
up=IQ['75%'] + 1.5 * (IQ['75%'] - IQ['25%'])

IQ.loc[:, 'low']=pd.Series(low)
IQ.loc[:, 'up']=pd.Series(up)

IQ=pd.DataFrame(IQ)
IQ.drop(['BodyFat', 'Density', 'Age'], axis=0, inplace=True)
print(IQ, IQ.shape)

for i in ['Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']:
    data[i]=data[i].apply(lambda x: 'NaN' if (x>IQ.loc[i, 'up']
                                              or x<IQ.loc[i, 'low']) else x)

data=data.replace(['NaN'], np.nan)
print(data)

# 剔除异常值后的缺失情况
na_ratio=data.isnull().sum()[data.isnull().sum()>0].sort_values(ascending=False)/len(data)
na_sum=data.isnull().sum().sort_values(ascending=False)
print(na_ratio, na_sum)

# fig,axes=plt.subplots(1,1,figsize=(12,6))
# # axes.grid(color='#909090',linestyle=':',linewidth=2)
# plt.xticks(rotation=90)
# sns.barplot(x=na_ratio.index,y=na_ratio,palette='coolwarm_r')
# plt.title('Missing Value Ratio',color=('#000000'),y=1.03)
# plt.tight_layout();
# plt.show()

# 剔除异常值后的缺失情况图
fig,axes=plt.subplots(1,1,figsize=(12,12))
sns.barplot(x=na_sum,y=na_sum.index,palette='brg')
for p in axes.patches:
            value = p.get_width()
            x = p.get_x() + p.get_width()+30
            y = p.get_y() + p.get_height()-.2
            axes.text(x, y, int(value),
                      ha="left",fontsize=11,
                      color='#000000',
                      bbox=dict(facecolor='#dddddd', edgecolor='black',boxstyle='round', linewidth=.5))

plt.title('Outliers Values',color=('#000000'),y=1.03)
plt.tight_layout();
plt.show()

# 用均值填充缺失值
for columns in ['Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']:
    data[columns].fillna(data[columns].mean(), inplace=True)

na_sum_new=data.isnull().sum().sort_values(ascending=False)
print(na_sum_new)

data.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_2.csv', index=None)