import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=15)



def plot_learning(result_dir, ax, smooth=1, interval=1, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)
fig, ax = plt.subplots(figsize=[5, 4])

plot_learning('result_loss/fedavg10/loss_test_centralized_width128_cifar10_all_data_1_dirichlet_niid2nn.npy',
              ax, label='loss_test_cen_width128', linestyle='-', color='tab:red')
plot_learning('result_loss/fedavg10/loss_test_width128_cifar10_all_data_1_dirichlet_niid2nn.npy',
              ax, label='loss_test_fed_width128', linestyle='--', color='tab:green')
plot_learning('result_loss/fedavg10/loss_test_centralized_width4096_cifar10_all_data_1_dirichlet_niid2nn.npy',
              ax, label='loss_test_cen_width4096', linestyle='-', color='tab:cyan')
plot_learning('result_loss/fedavg10/loss_test_width4096_cifar10_all_data_1_dirichlet_niid2nn.npy',
              ax, label='loss_test_fed_width4096', linestyle='--', color='tab:orange')

ax.set_xlim([0, 1000])
ax.set_ylim([1, 8])
ax.set_xticks(np.arange(0,1200,200))
ax.set_yticks([1,2,3,4,5,6,7,8])
ax.set_xlabel('Global Round')
ax.set_ylabel('Loss')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()
ax.legend(handlelength=2.3)
fig.tight_layout()
plt.show()