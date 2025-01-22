import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=15)

def plot_learning(result_dir, ax, smooth=10, interval=10, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')

    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)


fig, ax = plt.subplots(figsize=[5, 4])
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_dirichlet_niid2nn_8.npy',
              ax, label='fc1_niid', linestyle='--', color='tab:purple')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_dirichlet_niid2nn_16.npy',
              ax, label='fc2_niid', linestyle='--', color='tab:green')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_dirichlet_niid2nn_32.npy',
              ax, label='fc4_niid', linestyle='--', color='tab:cyan')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_dirichlet_niid2nn_128.npy',
              ax, label='fc16_niid', linestyle='--', color='tab:red')

plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_random_iid2nn_8.npy',
              ax, label='fc1_iid', linestyle='-', color='tab:purple')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_random_iid2nn_16.npy',
              ax, label='fc2_iid', linestyle='-', color='tab:green')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_random_iid2nn_32.npy',
              ax, label='fc4_iid', linestyle='-', color='tab:cyan')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_random_iid2nn_128.npy',
              ax, label='fc16_iid', linestyle='-', color='tab:red')

ax.set_xlim([0, 500])
ax.set_ylim([0.1, 1])
ax.set_xticks(np.arange(0,600,100))
ax.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
ax.set_xlabel('Global Round')
ax.set_ylabel('Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
legend = ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.19),
    fontsize=12,
    frameon=True,
    ncol=4,
    handlelength=1.5,
    columnspacing=0.8,
    handletextpad=0.5
)
legend.get_frame().set_alpha(0.8)
legend.get_frame().set_edgecolor('black')
fig.tight_layout()
plt.subplots_adjust(bottom=0.3)
plt.show()