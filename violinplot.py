#  -*- coding = utf-8 -*- 
#  @time2022/4/1623:52
#  Author:Ruanguojie

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.figure(figsize=(4.72, 3.15), dpi=300)  # figure大小(英寸)和分辨率
sns.set_theme(style="ticks", font='Times New Roman')

data = pd.read_csv('***.csv')
cmap1 = ['#3E61A5', '#9696D2', '#FFC697', '#FF655E', '#B83C3A']  # 自定义自己喜欢的颜色
cmap2 = ['#4A8FB9', '#92C0D8', '#D0E3F1', '#FAD9C6', '#ECA382']
cmap3 = ['#FFC4C6', '#97E1B0', '#C3D6FE', '#8ECBB9', '#E3AD7F']
cmap4 = ['#D0E1F1', '#98C6DE', '#4D99CA', '#1968AD', '#083674']

# 小提琴图
g = sns.violinplot(data=data, x="***", y="***",
                   linewidth=0.8,  # 轮廓线宽度
                   palette='Blues',  # 调色板，可以更改为自己定义的cmap
                   width=0.9,  # 宽度
                   inner='box',
                   )

'''
# 增强型箱线图
sns.boxenplot(data=data, x="Site", y="PNU",
              linewidth=1,  # 轮廓线宽度
              palette=cmap1,  # 调色板
              width=.6,  # 宽度
              ax=ax)
'''
# get_children 可以控制某元素的颜色
g.get_children()[11].set_color('white')  # 控制inner box的颜色
g.get_children()[13].set_color('white')
g.get_children()[15].set_color('white')
g.get_children()[17].set_color('white')
g.get_children()[19].set_color('white')

plt.setp(g.collections, edgecolor='black')  # 更改edgecolor的颜色，适用于boxenplot，violinplot
sns.despine(left=False, top=False, bottom=False, right=False)
g.set_xlabel('***', fontsize=12)  # 坐标轴标题和字体大小
g.set_ylabel('***', fontsize=12)
# g.ax.set_title('***')  # 标题
g.tick_params(labelsize=12, length=3, width=1)  # 坐标轴相关参数
plt.grid(linewidth=0.5, alpha=0.5, linestyle='--')  # 网格设置
plt.tight_layout()
plt.subplots_adjust(left=0.16)  # 让边框等长
plt.savefig('***.jpeg')  # 这句要放在plt.show()之前，否则保存为空白
plt.show()
