import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=15)
matplotlib.rc('font', family='sans-serif')
import matplotlib.ticker as ticker

def plot_learning(result_dir, ax, smooth=10, interval=10, **kwargs):
    data = np.load(result_dir, allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])
plot_learning('result_loss/fedavg9/loss_train_client_0_width128mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='$f_i$_width128', linestyle='-', color='tab:blue')
plot_learning('result_loss/fedavg9/linear_loss_train_client_0_width128mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='$f_i^{\\bf lin}$_width128', linestyle='--', color='tab:blue')
plot_learning('result_loss/fedavg9/loss_train_client_0_width4096mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='$f_i$_width4096', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg9/linear_loss_train_client_0_width4096mnist_all_data_1_linear_regression_niidlinear_regression.npy',
              ax, label='$f_i^{\\bf lin}$_width4096', linestyle='--', color='tab:red')
ax.set_yscale('log', base=10)

y_ticks = [10**i for i in range(-8, 3, 2)]
ax.set_yticks(y_ticks)
formatter = ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$')
ax.yaxis.set_major_formatter(formatter)


ax.set_xlim([0, 2500])
ax.set_ylim([10**-8, 10**2])
ax.set_xticks(np.arange(0, 3000, 500))
ax.set_xlabel('Total Round')
ax.set_ylabel('Loss')
ax.grid()
ax.legend(handlelength=2.3, bbox_to_anchor=(0.7, 0.89), loc='upper center')
fig.tight_layout()
plt.show()
