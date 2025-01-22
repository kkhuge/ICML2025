import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=17)
matplotlib.rc('font', family='sans-serif')

def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)
fig, ax = plt.subplots(figsize=[5,4])

plot_learning('result_output_differ/fedavg9/width_128_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='output_diff_fc16', linestyle='-', color='tab:blue')
plot_learning('result_output_differ/fedavg9/width_4096_diff_nonlinear_linearlinear_regressionmnist_all_data_1_linear_regression_niid.npy',
              ax, label='output_diff_fc512', linestyle='-', color='tab:red')
ax.set_yscale('log', base=10)

y_ticks = [10**(-i) for i in range(-1, 5)]
ax.set_yticks(y_ticks)
formatter = matplotlib.ticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$')
ax.yaxis.set_major_formatter(formatter)

ax.set_xlim([0, 500])
ax.set_ylim([10**-4, 10**1])
ax.set_xticks(np.arange(0,600,100))
ax.set_xlabel('Gloabl Round')
ax.set_ylabel(r'$\sqrt{||f^{\rm lin} - f||^2_2/|\mathcal{D}|}$')
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()