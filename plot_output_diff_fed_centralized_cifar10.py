import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=15)

def plot_learning(result_dir, ax, smooth=2, interval=2, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])

plot_learning('result_output_differ/fedavg6/output_width128_linear_regressioncifar10_all_data_1_linear_regression_niid.npy',
              ax, label='output_diff_width_128', linestyle='-', color='tab:blue')
plot_learning('result_output_differ/fedavg6/output_width4096_linear_regressioncifar10_all_data_1_linear_regression_niid.npy',
              ax, label='output_diff_width_4096', linestyle='-', color='tab:red')

ax.set_xlim([0, 500])
ax.set_ylim([0,2])
ax.set_xticks(np.arange(0,600,100))
ax.set_yticks([0,0.5,1,1.5,2])
ax.set_xlabel('Local Round')
ax.set_ylabel(r'$\sqrt{||f^{t\tau}_{centralized} - f^{t\tau}_{FedAvg}||^2_2/|\mathcal{D}|}$')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()