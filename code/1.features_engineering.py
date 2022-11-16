import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

data=pd.read_csv('data/bodyfat.csv')
pd.set_option('display.max_columns', None)
print(data.describe().T)

data['Weight'] = data['Weight'] * 0.4536
data['Height'] = data['Height'] * 2.54 * 0.01

data_stata=pd.DataFrame(data.describe().T)
print(data_stata)
data_stata.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_stata.csv')


print('Add BMI:', '\n', data.describe().T)

print(data.columns)

# Numerical features distribution 1
nem_col=['Density', 'Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip']
plt.figure(figsize=(12, 8))
i = 1
plt.rcParams.update({'font.size': 15})
for col in nem_col:
    ax = plt.subplot(3, 3, i)
    ax = sns.kdeplot(data=data[col], bw=0.5, color="Red", legend=False, shade=True)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    i += 1
    plt.tight_layout();
plt.show()

# Numerical features distribution 2
nem_col=['Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist']
plt.figure(figsize=(12, 8))
i = 1
plt.rcParams.update({'font.size': 15})
for col in nem_col:
    ax = plt.subplot(3, 3, i)
    ax = sns.kdeplot(data=data[col], bw=0.5, color="Red", legend=False, shade=True)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    i += 1
    plt.tight_layout();
plt.show()

data.to_csv(path_or_buf=r'C:/Users/Gealen/PycharmProjects/pythonProject/BodyFat/data/data_1.csv',index = None)
