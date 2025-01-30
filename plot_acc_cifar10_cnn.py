import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font', size=15)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes



def plot_learning(result_dir, ax, smooth=100, interval=100, **kwargs):
    data = np.load(result_dir,allow_pickle=True)
    if smooth > 1:
        data = np.convolve(data, np.ones(smooth) / smooth, mode='valid')
    mean = data[::interval]
    episode = np.arange(len(data))[::interval]
    ax.plot(episode, mean, **kwargs)


fig, ax = plt.subplots(figsize=[5, 4])
#train
plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet_1.npy',
              ax, label='CNN1_iid', linestyle='-', color='tab:purple')
plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet_2.npy',
              ax, label='CNN2_iid', linestyle='-', color='tab:green')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet_4.npy',
#               ax, label='CNN4_iid', linestyle='-', color='tab:cyan')
plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet_8.npy',
              ax, label='CNN8_iid', linestyle='-', color='tab:cyan')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet_16.npy',
#               ax, label='CNN16_iid', linestyle='-', color='tab:orange')
plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet_32.npy',
              ax, label='CNN32_iid', linestyle='-', color='tab:red')

plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet_1.npy',
              ax, label='CNN1_niid', linestyle='--', color='tab:purple')
plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet_2.npy',
              ax, label='CNN2_niid', linestyle='--', color='tab:green')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet_4.npy',
#               ax, label='CNN4_niid', linestyle='--', color='tab:cyan')
plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet_8.npy',
              ax, label='CNN8_niid', linestyle='--', color='tab:cyan')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet_16.npy',
#               ax, label='CNN16_niid', linestyle='--', color='tab:orange')
plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet_32.npy',
              ax, label='CNN32_niid', linestyle='--', color='tab:red')

# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet1.npy',
#               ax, label='CNN1_iid', linestyle='-', color='tab:purple')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet2.npy',
#               ax, label='CNN2_iid', linestyle='-', color='tab:blue')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet4.npy',
#               ax, label='CNN4_iid', linestyle='-', color='tab:cyan')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet8.npy',
#               ax, label='CNN8_iid', linestyle='-', color='tab:green')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet16.npy',
#               ax, label='CNN16_iid', linestyle='-', color='tab:orange')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_random_iidlenet32.npy',
#               ax, label='CNN32_iid', linestyle='-', color='tab:red')
#
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet1.npy',
#               ax, label='CNN1_niid', linestyle='--', color='tab:purple')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet2.npy',
#               ax, label='CNN2_niid', linestyle='--', color='tab:blue')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet4.npy',
#               ax, label='CNN4_niid', linestyle='--', color='tab:cyan')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet8.npy',
#               ax, label='CNN8_niid', linestyle='--', color='tab:green')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet16.npy',
#               ax, label='CNN16_niid', linestyle='--', color='tab:orange')
# plot_learning('result_acc/fedavg5/acc_testcifar10_all_data_1_dirichlet_niidlenet32.npy',
#               ax, label='CNN32_niid', linestyle='--', color='tab:red')












ax.set_xlim([0, 10000])
ax.set_ylim([0.1, 0.9])
ax.set_xticks(np.arange(0,12000,2000))
ax.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6, 0.7, 0.8,0.9])
ax.set_xlabel('Global Round')
ax.set_ylabel('Test Accuracy')
ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid()

legend = ax.legend(
    loc='upper center',         
    bbox_to_anchor=(0.5, -0.19),  
    fontsize=12,              
    frameon=True,              
    ncol=4,                    
    handlelength=1.5,            
    columnspacing=0.8,          
    handletextpad=0.5          
)


legend.get_frame().set_alpha(0.8)
legend.get_frame().set_edgecolor('black')

fig.tight_layout()
plt.subplots_adjust(bottom=0.3)  
plt.show()
