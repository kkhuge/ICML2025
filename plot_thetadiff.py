import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=17)
matplotlib.rc('font', family='sans-serif')

def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir, allow_pickle=True)

    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])
plot_learning('result_theta/fedavg4/width_128linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='width_128', linestyle='-', color='tab:purple')
plot_learning('result_theta/fedavg4/width_256linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='width_256', linestyle='-', color='tab:blue')
plot_learning('result_theta/fedavg4/width_512linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='width_512', linestyle='-', color='tab:cyan')
plot_learning('result_theta/fedavg4/width_1024linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='width_1024', linestyle='-', color='tab:green')
plot_learning('result_theta/fedavg4/width_2048linear_regressionmnist_all_data_1_linear_regression_niid.npy',ax, label='width_2048', linestyle='-', color='tab:orange')

ax.set_yscale('log', base=2)
y_ticks = [2**(-i) for i in range(0, 10, 2)]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'$2^{{-{i}}}$' for i in range(0, 10, 2)])
ax.set_xticks(np.arange(0, 600, 100))
ax.set_xlim([0, 500])
ax.set_ylim([2**-8, 1])  # 根据负指数设置y轴范围
ax.set_xlabel('Global Round')
ax.set_ylabel(r'$\|\Theta^{t\tau} - \Theta^0\|_F / \|\Theta^0\|_F$')
ax.grid(True, which="both", ls="--")
legend = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.19),
    fontsize=14,
    frameon=True,
    ncol=3,
    handlelength=2,
    columnspacing=0.8,
    handletextpad=0.5
)
legend.get_frame().set_alpha(0.8)
legend.get_frame().set_edgecolor('black')
fig.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.show()