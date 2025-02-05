from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import LrdWorker
from torch.optim import SGD
import numpy as np
import torch
import os


criterion = torch.nn.CrossEntropyLoss()


loss_dir = "result_loss/fedavg5"
acc_dir = "result_acc/fedavg5"

if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)



class FedAvg5Trainer(BaseTrainer):
    """
    Original Scheme
    """
    def __init__(self, options, dataset):
        self.theta_0 = 0
        self.error_train = []
        self.loss_list_train = []
        self.acc_list_train = []
        self.loss_list_test = []
        self.acc_list_test = []
        self.theta = []
        self.diff_nonlinear_linear = []
        self.weight_change = []
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.tau = options['num_epoch']
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0001)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = LrdWorker(model, self.optimizer, options)
        super(FedAvg5Trainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        self.latest_model = self.worker.get_flat_model_params().detach()
        for round_i in range(self.num_round):
            loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
            self.acc_list_test.append(accuracy_test)
            self.loss_list_test.append(loss_test)
            selected_clients = self.select_clients(seed=round_i)
            solns, stats = self.local_train(round_i, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)
            self.latest_model = self.aggregate(solns)
        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.acc_list_test.append(accuracy_test)
        self.loss_list_test.append(loss_test)
        #train the large cnn and the large fully connected network which describe in fig.2
        np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_large8', self.loss_list_test)
        np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_large8', self.acc_list_test)
        # #train the family of fully connected network
        # np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_fc1', self.loss_list_test)
        # np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_fc1', self.acc_list_test)
        # #train the family of cnn
        # np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_1', self.loss_list_test)
        # np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_1', self.acc_list_test)
        # #train the family of resnet
        # np.save(loss_dir + '/loss_test' + self.dataset + self.model, self.loss_list_test)
        # np.save(acc_dir + '/acc_test' + self.dataset + self.model, self.acc_list_test)



        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()
