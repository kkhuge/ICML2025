import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True  # 支持 LaTeX 格式
matplotlib.rc('font', size=15)  # 设置字体
matplotlib.rc('font', family='sans-serif')

def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir, allow_pickle=True)

    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    mean = data[::interval]
    episode = np.arange(len(data))[::interval] ##本地训练轮数tau=8
    ax.plot(episode, mean, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])



plot_learning('result_theta/fedavg12/client_0_width_128linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='FNN16', linestyle='-', color='tab:purple')
plot_learning('result_theta/fedavg12/client_0_width_256linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='FNN32', linestyle='-', color='tab:blue')
plot_learning('result_theta/fedavg12/client_0_width_512linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='FNN64', linestyle='-', color='tab:cyan')
plot_learning('result_theta/fedavg12/client_0_width_1024linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='FNN128', linestyle='-', color='tab:green')
plot_learning('result_theta/fedavg12/client_0_width_2048linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='FNN256', linestyle='-', color='tab:orange')


#

# 设置横轴为对数刻度，底数为 2
ax.set_yscale('log', base=2)



# 设置纵轴刻度和标签，y 轴用2的负幂
y_ticks = [2**(-i) for i in range(0, 10, 2)]  # 生成负的指数次刻度
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'$2^{{-{i}}}$' for i in range(0, 10, 2)])  # 格式化标签为2的负幂

# 其他设置
ax.set_xticks(np.arange(0, 3000, 500))
ax.set_xlim([0, 2500])
ax.set_ylim([2**-8, 1])  # 根据负指数设置y轴范围
ax.set_xlabel('Total Iteration')
ax.set_ylabel(r'$\|\Theta_i^{t\tau+r} - \Theta_i^0\|_F / \|\Theta_i^0\|_F$')
ax.grid(True, which="both", ls="--")
# 创建图例，放置在图表下方
legend = ax.legend(
    loc='upper center',          # 图例位置
    bbox_to_anchor=(0.5, -0.19),  # 图例放在轴下方
    fontsize=12,                 # 图例字体大小
    frameon=True,                # 显示边框
    ncol=3,                      # 图例分为三列
    handlelength=2,            # 图例线条长度
    columnspacing=0.8,           # 列间距
    handletextpad=0.5            # 线条与文本的间距
)

# 调整图例边框样式
legend.get_frame().set_alpha(0.8)
legend.get_frame().set_edgecolor('black')

fig.tight_layout()
plt.subplots_adjust(bottom=0.3)  # 增加图表与底部图例的间距
plt.show()