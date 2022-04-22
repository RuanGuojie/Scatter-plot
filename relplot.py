#  -*- coding = utf-8 -*- 
#  @time2022/4/1621:04
#  Author:Ruanguojie

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style='ticks', font='Times New Roman', font_scale=1)  # 控制主题和字体，ticks主题为加小刻度线
# 查看更多 seaborn API reference： http://seaborn.pydata.org/api.html
data = pd.read_csv('***.csv')  # read your data
cmap = sns.cubehelix_palette(n_colors=5, rot=-.2, gamma=0.6, as_cmap=False)
g = sns.relplot(
    data=data,
    x="***", y="***",
    hue="***",
    size="***",
    size_order=['***', '***', '***'],
    palette=cmap,
    sizes=(5, 40),
    legend=False,
)

sns.despine(left=False, bottom=False, top=False, right=False)  # 隐藏主坐标轴，True为隐藏，False为显示
g.set_xlabels('***', fontsize=12)  # 坐标轴标题和字体大小
g.set_ylabels('***', fontsize=12)
# g.ax.set_title('***')  # 标题
g.ax.set_xlim([0, 0.5])  # 横坐标范围,不设置的话左下角会有空隙
g.ax.set_ylim([0, 400])  # 纵坐标范围,根据需要更改
g.figure.set_size_inches(4.72, 3.15)
# relplot是一种FacetGrid, 无法通过plt.figure(figsize)控制大小
plt.tick_params(labelsize=12, length=3, width=1)  # 坐标轴相关参数
plt.tight_layout()
plt.subplots_adjust(left=0.15)  # 让图片边框等长，否则figure大小一致，但边框不等长
g.savefig('***.jpeg', dpi=300)  # 更改文件名，格式和分辨率
plt.show()  # 保存的图片才是真实效果，plt.show()只是辅助
