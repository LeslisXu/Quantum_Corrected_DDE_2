import matplotlib.pyplot as plt
import numpy as np

# 设置科研绘图风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 16,
    'figure.figsize': (6, 4),
    'figure.dpi': 300,
    'axes.linewidth': 1.2,
    'grid.linestyle': '--',
    'grid.alpha': 0.5
})

# 数据准备
# voltage = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# S1 = [1, 12, 14, 16, 16, 16, 18, 21, 21]
# S2 = [1, 12, 14, 16, 16, 16, 18, 21, 22]
# S3 = [1, 12, 15, 16, 16, 17, 18, 21, 22]

voltage = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
S1      = [1,2,3,4,6,8,10,11,12]
S2      = [1,2,4,4,7,8,10,11,13]
S3      = [1,2,4,4,7,9,11,12,12]

# 创建图形
fig, ax = plt.subplots()

# 绘制三条曲线
lines = [
    ('S1', '-', 'o', "#E7879F"),    # 实线 + 圆形
    ('S2', '--', 's', '#456990'),   # 虚线 + 方形
    ('S3', '-.', '^', '#48C0AA')    # 点划线 + 三角形
]

for idx, (label, linestyle, marker, color) in enumerate(lines):
    ydata = [S1, S2, S3][idx]
    ax.plot(voltage, ydata,
            linestyle=linestyle,
            linewidth=1.5,
            marker=marker,
            markersize=8,
            markerfacecolor='white',
            markeredgewidth=1.5,
            color=color,
            label=label)

# 坐标轴设置
ax.set_xlabel('Applied Voltage (V)', fontweight='bold')
ax.set_ylabel('Iteration Number', fontweight='bold')
ax.set_xticks(np.arange(0, 1.0, 0.1))
ax.set_yticks(np.arange(0, 25, 5))
ax.set_xlim(-0.05, 0.95)
ax.set_ylim(0, 24)

# 图例设置
ax.legend(
    frameon=True,
    loc='upper left',
    edgecolor='black',
    facecolor='white',
    framealpha=0.8
)

# 优化布局
plt.tight_layout()

# 保存图片
plt.savefig('voltage_iteration_newton.png', bbox_inches='tight', dpi = 300)
# plt.show()
