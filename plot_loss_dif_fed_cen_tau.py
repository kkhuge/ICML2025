import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=17)
tau = 5 #2

def plot_learning(result_dir, ax, smooth=1, interval=1, max_points=None, **kwargs):
    data = np.load(result_dir, allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    if max_points is not None:
        data = data[:max_points]
    data = data[::interval]
    episode = np.arange(len(data))
    ax.plot(episode, data, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])

plot_learning('result_loss/fedavg11/loss_train_centralized_width4096_tau5_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
    ax, interval=tau, label='loss_train_cen_fc512', linestyle='-', color='tab:cyan')
plot_learning('result_loss/fedavg11/loss_train_width4096_tau5_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
    ax, max_points=100, label='loss_train_fed_fc512', linestyle='--', color='tab:red')
plot_learning('result_loss/fedavg11/loss_test_centralized_width4096_tau5_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
    ax, interval=tau, label='loss_test_cen_fc512', linestyle='-', color='tab:orange')
plot_learning('result_loss/fedavg11/loss_test_width4096_tau5_cifar10_all_data_1_linear_regression_niidlinear_regression.npy',
    ax, max_points=100, label='loss_test_fed_fc512', linestyle='--', color='tab:green')

ax.set_xlim([0, 100])
ax.set_ylim([0, 3])
ax.set_xticks(np.arange(0, 110, 20))
ax.set_xticklabels((np.arange(0, 110, 20) * tau).astype(int))
ax.set_yticks([0, 0.5, 1, 1.5, 2, 3])
ax.set_xlabel('Total Iteration')
ax.set_ylabel('Loss')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()
