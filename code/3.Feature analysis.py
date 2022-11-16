import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data=pd.read_csv('data/data_2.csv')
pd.set_option('display.max_columns', None)
print(data.shape, '\n', data.isnull().sum())

### 构造新特征BMI
data['BMI']= data['Weight'] / np.power(data['Height'], 2)
print(data.describe().T, '\n', data.head(10), '\n', data.shape)
#
# #BMI的分布
# data.BMI.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = 'hist')
# data.BMI.plot(kind = 'kde', color = 'red', label = 'kde')
# plt.xlabel('BMI',fontsize=15)
# plt.ylabel('Frequency',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title('Before Box-Cox', fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
# # Weight的分布
# data.Weight.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = 'hist')
# data.Weight.plot(kind = 'kde', color = 'red', label = 'kde')
# plt.xlabel('Weight', fontsize=15)
# plt.ylabel('Frequency',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title('Before Box-Cox', fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
# # Chest的分布
# data.Chest.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = 'hist')
# data.Chest.plot(kind = 'kde', color = 'red', label = 'kde')
# plt.xlabel('Chest',fontsize=15)
# plt.ylabel('Frequency',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title('Before Box-Cox', fontsize=15)
# plt.legend(fontsize=15)
# plt.show()

## 分析各特征与BodyFat的相关性
# data_col=['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist', 'BMI']
# fig,axes=plt.subplots(ncols=5,nrows=3,figsize=(12,12))
# for i, feature in enumerate(data[data_col]):
#     row = int(i/5)
#     col = i%5
#     sns.regplot(x=data[feature],y=data['BodyFat'], ax=axes[row][col],ci=0,line_kws={'color':'#000000','linewidth':2},marker='o')
# plt.suptitle('New Features',y=1,size=20)
# # data[data].iloc[:, i]
# plt.tight_layout()
# plt.show()

# 相关性热力图
plt.rcParams['axes.unicode_minus']=False
corr=data.corr()
print(corr)
mask=np.triu(np.ones_like(corr, dtype=np.bool))
fig=plt.figure(figsize=(10, 12))
ax=sns.heatmap(corr, mask=mask, fmt=".2f", cmap='gist_heat', cbar_kws={"shrink": .8},
            annot=True, linewidths=1, annot_kws={"fontsize":15})
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize=15, rotation=30)
plt.yticks(fontsize=15, rotation=30)
plt.show()
#
### 查看各特征的正态性、偏度
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

# data.drop('BodyFat', axis=1, inplace=True)
skewed=dict(data.skew().sort_values(ascending=False))
skewed_PD_Before=pd.DataFrame(data=skewed.values(), index=skewed.keys(), columns=['Skewed Values'])
pd.set_option('display.max_columns', None)
print('skewed_PD_Before:', skewed_PD_Before)

from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

for i in skewed.keys():
       if skewed[i]<.2:
            continue
       else:
            data[i] = boxcox1p(data[i], boxcox_normmax(data[i] + 1))

skewed=dict(data.skew().sort_values(ascending=False))
skewed_PD_After=pd.DataFrame(data=skewed.values(), index=skewed.keys(), columns=['Skewed Values'])
print('skewed_PD_After:', skewed_PD_After)
### 对偏度大于0.2的特征，进行boxcox1p转换。

# 比较变换后的特征分布
# # BMI
# data.BMI.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = 'hist')
# data.BMI.plot(kind = 'kde', color = 'red', label = 'kde')
# plt.xlabel('BMI',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.ylabel('Frequency',fontsize=15)
# plt.title('After Box-Cox', fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
# # Weight
# data.Weight.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = 'hist')
# data.Weight.plot(kind = 'kde', color = 'red', label = 'kde')
# plt.xlabel('Weight', fontsize=15)
# plt.ylabel('Frequency', fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title('After Box-Cox', fontsize=15)
# plt.legend(fontsize=15)
# plt.show()
#
# # Chest
# data.Chest.plot(kind = 'hist', bins = 20, color = 'steelblue', edgecolor = 'black', density = True, label = 'hist')
# data.Chest.plot(kind = 'kde', color = 'red', label = 'kde')
# plt.xlabel('Chest',fontsize=15)
# plt.ylabel('Frequency',fontsize=15)
# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)
# plt.title('After Box-Cox', fontsize=15)
# plt.legend(fontsize=15)
# plt.show()


from scipy import stats
from scipy.stats import norm
print('skewed of BodyFat:', data['BodyFat'].skew())
#
# plt.figure(figsize=(12,6))
# stats.probplot(data['BodyFat'],plot=plt);
# plt.figure(figsize=(12,6))
# mu, sigma = norm.fit(data['BodyFat'])
# sns.distplot(data['BodyFat'],fit=norm,color='b',rug=True,kde_kws={'shade':True,'color':'b','alpha':.2})
# plt.legend(['$\mu=$ {:.3f} and $\sigma=$ {:.3f}'.format(mu, sigma)],fontsize=14)
# plt.title('BodyFat',size=20)
# plt.tight_layout();
# plt.show()

## 分析各特征与BodyFat的相关性
corrs=data.drop('BodyFat',axis=1).corrwith(data['BodyFat']).sort_values(ascending=False)
print('corrs with BodyFat:', '\n', corrs)

fig,axes=plt.subplots(1,1,figsize=(12,6))
axes.axhline(corrs[corrs>0].mean(), ls=':',color='black',linewidth=2)
axes.text(10,corrs[corrs>0].mean()+.015, "Average = {:.3f}".format(corrs[corrs>0].mean()),color='black',size=14)
axes.axhline(corrs[corrs<0].mean(), ls=':',color='black',linewidth=2)
axes.text(10,corrs[corrs<0].mean()+.015, "Average = {:.3f}".format(corrs[corrs<0].mean()),color='black',size=14)
sns.barplot(y=corrs,x=corrs.index,palette='Spectral')
plt.title('Correlation of BodyFat to other Features',size=20,color='black',y=1.03)
plt.xticks(rotation=90)
for p in axes.patches:
            value = p.get_height()
            if value <=.5:
                continue
            x = p.get_x() + p.get_width()-.9
            y = p.get_y() + p.get_height()+(.02*value)
            axes.text(x, y, str(value)[1:5], ha="left",fontsize=12,color='#000000')
plt.tight_layout();
plt.show()



data.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_clean.csv',index = None)