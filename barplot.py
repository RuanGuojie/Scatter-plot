#  -*- coding = utf-8 -*- 
#  @time2022/4/1719:03
#  Author:Ruanguojie

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()  # 用于创建双y轴barplot
plt.figure(figsize=(4.72, 3.54), dpi=300)
sns.set(style='ticks', font='Times New Roman', font_scale=1)

data = pd.read_csv('***.csv')
g = sns.barplot(x="***",
                y="***",
                hue="***",
                data=data,
                capsize=.05,
                errcolor='black',  # 误差线颜色
                errwidth=1,  # 误差线宽度
                palette="Blues_r",
                )

sns.despine(left=False, top=False, bottom=False, right=False)
g.set_xlabel('***', fontsize=12)  # 坐标轴标题和字体大小
g.set_ylabel('***', fontsize=12)
# g.ax.set_title('***')  # 标题
g.tick_params(labelsize=12, length=3, width=1)  # 坐标轴相关参数
g.grid(linewidth=0.5, alpha=0.5, linestyle='--')  # 网格设置
g.legend(frameon=False, fontsize=12, bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.savefig('***.jpeg')
plt.show()

