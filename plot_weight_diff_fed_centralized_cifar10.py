import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=17)

def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir, allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)

fig, ax = plt.subplots(figsize=[5, 4])

plot_learning('result_weight_change/fedavg6/weight_change_width128_linear_regressioncifar10_all_data_1_linear_regression_niid.npy',
              ax, label='fc16', linestyle='-', color='tab:blue')
plot_learning('result_weight_change/fedavg6/weight_change_width4096_linear_regressioncifar10_all_data_1_linear_regression_niid.npy',
              ax, label='fc512', linestyle='-', color='tab:red')

ax.set_yscale('log', base=10)
y_ticks = [10**i for i in range(-9, -2)]
ax.set_yticks(y_ticks)
ax.set_yticklabels([f'$10^{i}$' for i in range(-9, -2)])
ax.set_xticks(np.arange(0, 600, 100))
ax.set_xlim([0, 500])
ax.set_ylim([10**(-9), 10**(-3)])
ax.set_xlabel('Global Round')
ax.set_ylabel(r'$\sqrt{||w_{\rm cen}^{t\tau} - w_{\rm fed}^{t\tau}||^2_2/len(w_{\rm fed}^{t\tau})}$')
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()
