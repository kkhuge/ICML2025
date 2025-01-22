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

plot_learning('result_loss/fedavg6/loss_test_centralized_width4096_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='loss_test_cen_fc512', linestyle='-', color='tab:orange')
plot_learning('result_loss/fedavg6/loss_test_width4096_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='loss_test_fed_fc512', linestyle='--', color='tab:green')
plot_learning('result_loss/fedavg6/loss_train_centralized_width4096_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='loss_train_cen_fc512', linestyle='-', color='tab:cyan')
plot_learning('result_loss/fedavg6/loss_train_width4096_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='loss_train_fed_fc512', linestyle='--', color='tab:red')



ax.set_xlim([0, 500])
ax.set_ylim([0, 3])
ax.set_xticks(np.arange(0,600,100))
ax.set_yticks([0, 0.5,1,1.5,2,3])
ax.set_xlabel('Global Round')
ax.set_ylabel('Loss')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()