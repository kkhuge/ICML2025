import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=12)



def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])


plot_learning('result_acc/fedavg6/acc_test_centralized_mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='acc_test_centralized_learning', linestyle='--', color='tab:red')
plot_learning('result_acc/fedavg6/acc_train_centralized_mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='acc_train_centralized_learning', linestyle='-', color='tab:red')
plot_learning('result_acc/fedavg6/acc_test_mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='acc_test_fedavg', linestyle='--', color='tab:blue')
plot_learning('result_acc/fedavg6/acc_train_mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='acc_train_fedavg', linestyle='-', color='tab:blue')



ax.set_xlim([0, 500])
ax.set_ylim([0, 1])
ax.set_xticks(np.arange(0,500,100))
ax.set_yticks([0,0.2,0.4,0.6,0.8,0.9, 1])
ax.set_xlabel('Round')
ax.set_ylabel('Accurancy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()