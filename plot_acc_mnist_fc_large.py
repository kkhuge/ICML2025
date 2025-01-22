import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=17)


def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])

plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_dirichlet_niid2nn_large8.npy',
              ax, label='fc1_niid', linestyle='-', color='tab:green')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_random_iid2nn_large8.npy',
              ax, label='fc1_iid', linestyle='-', color='tab:purple')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_dirichlet_niid2nn_large256.npy',
              ax, label='fc32_niid', linestyle='-', color='tab:cyan')
plot_learning('result_acc/fedavg5/acc_testmnist_all_data_1_random_iid2nn_large256.npy',
              ax, label='fc32_iid', linestyle='-', color='tab:red')

ax.set_xlim([0, 10000])
ax.set_ylim([0, 1])
ax.set_xticks(np.arange(0,12000,2000))
ax.set_yticks([0,0.2,0.4,0.6,0.8,0.9, 1])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accurancy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()