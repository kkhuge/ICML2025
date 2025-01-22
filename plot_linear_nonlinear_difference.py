import numpy as np
import pandas as pd
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

plot_learning('result_output_differ/fedavg4/width_128_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='diff_train_width128_niid', linestyle='--', color='tab:purple')
plot_learning('result_output_differ/fedavg4/width_256_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='diff_train_width256_niid', linestyle='--', color='tab:blue')
plot_learning('result_output_differ/fedavg4/width_512_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='diff_train_width512_niid', linestyle='--', color='tab:cyan')
plot_learning('result_output_differ/fedavg4/width_1024_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='diff_train_width1024_niid', linestyle='--', color='tab:green')
plot_learning('result_output_differ/fedavg4/width_2048_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='diff_train_width2048_niid', linestyle='--', color='tab:orange')
plot_learning('result_output_differ/fedavg4/width_4096_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='diff_train_width4096_niid', linestyle='--', color='tab:red')


ax.set_yscale('log', base=10)
y_ticks = [10**i for i in range(-5,1)]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'$10^{i}$' for i in range(-5,1)])
ax.set_xticks(np.arange(0,400,100))
ax.set_xlim([0,400])
ax.set_ylim([10**(-5), 10**(0)])
ax.set_xlabel('Round')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
ax.set_title('RMES with linear network and network')
fig.tight_layout()
plt.show()