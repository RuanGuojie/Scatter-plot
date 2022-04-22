#  -*- coding = utf-8 -*- 
#  @time2022/4/1511:21
#  Author:Ruanguojie

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# 画图模块
import matplotlib.pyplot as plt
import seaborn as sns
import palettable

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
# 读取数据
dataset = pd.read_csv('***.csv')
# 打乱数据集：
dataset = dataset.sample(frac=1.0, random_state=42)
data = dataset.reset_index(drop=True)

print(dataset.head())
print(dataset.describe())

target = dataset.pop('***')  # 标记放在最后一行

# Split into train/test datasets
# split data into train test
X_train, X_test, y_train, y_test = train_test_split(dataset.values.astype(np.float32),
                                                    target.values.reshape(-1, 1).astype(np.float32),
                                                    test_size=.2, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
# Standard Normalization Preprocess
# normalize data to 0 mean and unit std
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

param_grid_RF = [{
    'n_estimators': [],
    'max_depth': [],
    'min_samples_split':[],
    'min_samples_leaf':[]
    }]
RF = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_RF, cv=5, n_jobs=-1)
RF.fit(X_train_scaled, np.ravel(y_train))
final_model_RF = RF.best_estimator_
print(RF.best_params_)

# 保存模型
import joblib
# save
joblib.dump(final_model_RF, '***.pkl')  # 更改文件名
# restore
regressor = joblib.load('***.pkl')

y_pred_RF = regressor.predict(X_test_scaled)  # 在测试集评估模型
RF_train_score = regressor.score(X_train_scaled, np.ravel(y_train))
RF_test_score = r2_score(np.ravel(y_test), y_pred_RF)
RF_test_MSE = MSE(np.ravel(y_test), y_pred_RF)
RMSE_test = RF_test_MSE ** 0.5   # RMSE

print("r^2 of RF regression on training data: %.4f" % RF_train_score)
print("r^2 of RF regression on test data: %.4f" % RF_test_score)
print("RMSE of RF regression on test data: %.4f" % RMSE_test)


# 散点图1：seaborn-regplot
# ax = plt.axes()
# plt.style.use('ggplot')  # 绘图风格
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 中文编码与负号的正常显示
plt.figure(figsize=(4.72, 3.15), dpi=500)  # 画布大小(英寸）和分辨率
ax = sns.regplot(x=y_test, y=y_pred_RF, ci=95, scatter_kws=dict(linewidth=0.7, edgecolor='white'))

plt.plot([0, y_test.max()],
         [0, y_test.max()],  # 横纵坐标刻度需要保持一致
         '--',
         linewidth=2,
         c='pink')
# plt.title('***', fontsize=15)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)  # 隐藏顶部和右侧的实线
# ax.spines['left'].set_position(('outward', 5))
# ax.spines['bottom'].set_position(('outward', 5))  # 偏移axis
plt.xlabel('***', fontsize=10)
plt.ylabel('***', fontsize=10)  # 横纵坐标轴标题
plt.title('***', fontsize=10)  # 图表标题
plt.tight_layout()
plt.subplots_adjust(left=0.15)
plt.savefig('***.jpeg')
plt.show()


# 散点图2：matplotlib-hexbin
plt.figure(figsize=(4.72, 3.15), dpi=500)  # 画布大小和分辨率
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 中文编码与负号的正常显示
ax = plt.axes()
TK = plt.gca()
TK.spines['bottom'].set_linewidth(0.5)  # 控制边框的线宽，因为matpltlib不像seaborn经过高级封装，很多元素需要自己调整
TK.spines['left'].set_linewidth(0.5)
TK.spines['top'].set_linewidth(0.5)
TK.spines['right'].set_linewidth(0.5)

# 使用palettable配色
cmap = palettable.colorbrewer.sequential.Blues_9.mpl_colormap
plt.hexbin(
    np.ravel(y_test),
    y_pred_RF,
    gridsize=20,  # gridsize越大，hex越多
    mincnt=1,
    cmap=cmap,
    edgecolors='white',
    linewidths=0)
plt.xlim(0, y_test.max())  # 横坐标刻度范围
plt.ylim(0, y_test.max())  # 纵坐标刻度范围, 两个刻度要一致，不然1：1线会错误！！
plt.locator_params(nbins=5)  # 坐标刻度数量
plt.tick_params(labelsize=10, width=0.5)  # 坐标轴刻度
plt.grid(ls='--', lw=0.5)  # 添加网格线
ax.set_axisbelow(True)  # 网格线置于底层
ax.plot((0, 1), (0, 1), transform=ax.transAxes,
        ls='-', c='gray', linewidth=0.5)  # 添加1：1线
cb = plt.colorbar()  # 添加图例 colorbar
# cb.set_label('Number of scatters', fontsize=10)  # 图例标记
cb.ax.tick_params(labelsize=8, width=0.5)  # 设置图例刻度的字体大小
cb.outline.set_linewidth(0.5)  # 图例外框
# cb.set_ticks(np.linspace(1, 10, 10))
plt.xlabel('***', fontsize=10)  # 横坐标标题
plt.ylabel('***', fontsize=10)  # 纵坐标轴标题
plt.title('***', fontsize=10)  # 图表标题
# 选择添加文本框
'''
text = plt.text(
    x=0.5,   # 注意更改
    y=1.5,
    s='R$\mathregular{^2}$ = %.4f' %
    RF_test_score +
    '\n' +
    'RMSE = %.4f' %   # 注意RMSE的单位
    RMSE_test,
    fontsize=8,
    bbox={
        'facecolor': 'white',
        'edgecolor': 'white',
        'alpha': 0.5,
        'pad': 0.5})
'''
plt.tight_layout()
plt.subplots_adjust(left=0.15)
plt.savefig('***.jpeg')
plt.show()
