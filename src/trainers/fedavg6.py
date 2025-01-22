from src.trainers.base import BaseTrainer
from src.models.model import Linear_Regression
from src.models.worker import MSEWorker
from torch.optim import SGD
import numpy as np
import torch
import os
import torch.nn as nn
from src.utils.torch_utils import get_flat_params_from
from src.utils.torch_utils import set_seed




weight_change_dir = "result_weight_change/fedavg6"
output_dir = "result_output_differ/fedavg6"
loss_dir = "result_loss/fedavg6"
acc_dir = "result_acc/fedavg6"


if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(weight_change_dir):
    os.makedirs(weight_change_dir)


class FedAvg6Trainer(BaseTrainer):
    def __init__(self, options, dataset):
        self.loss_list_train = []
        self.loss_list_train_centralized = []
        self.acc_list_train = []
        self.acc_list_train_centralized = []
        self.loss_list_test = []
        self.loss_list_test_centralized = []
        self.acc_list_test = []
        self.acc_list_test_centralized = []
        self.diff_fed_centralized = []
        self.weight_diff_fed_centralized = []
        seed = 12345
        set_seed(seed)
        if options['dataset'] == 'cifar10_all_data_1_linear_regression_niid' or options['dataset'] == 'cifar10_all_data_1_linear_regression_iid':
            model = Linear_Regression(32*32*3,1)
        else:
            model = Linear_Regression(28*28,1)
        self.move_model_to_gpu(model, options)
        seed = 12345
        set_seed(seed)
        if options['dataset'] == 'cifar10_all_data_1_linear_regression_niid' or options['dataset']=='cifar10_all_data_1_linear_regression_iid':
            self.centralized_model = Linear_Regression(32*32*3,1)
        else:
            self.centralized_model = Linear_Regression(28*28,1)
        self.move_model_to_gpu(self.centralized_model, options)
        self.required_accuracy = options['psi']
        self.tau = options['num_epoch']
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0005)
        self.centralized_optimizer = SGD(self.centralized_model.parameters(), lr=options['lr'], weight_decay=0.0005)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = MSEWorker(model, self.optimizer,options)
        super(FedAvg6Trainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        self.latest_model = self.worker.get_flat_model_params().detach()
        self.last_centralized_model = self.latest_model.clone()
        for round_i in range(self.num_round):
            out = self.get_out()
            centralized_out = self.get_centralized_out(self.centralized_model)

            weight_diff = torch.sqrt(torch.mean((self.latest_model - self.last_centralized_model) ** 2)).item() #RMSE
            self.weight_diff_fed_centralized.append(weight_diff)


            differ = torch.sqrt(torch.mean((out - centralized_out) ** 2)).item()#RMSE
            self.diff_fed_centralized.append(differ)

            _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
            self.loss_list_train.append(loss)
            self.acc_list_train.append(accuracy)
            accuracy_centralized, loss_centralized = self.test_latest_centralized_model_on_traindata(self.centralized_model)
            self.loss_list_train_centralized.append(loss_centralized)
            self.acc_list_train_centralized.append(accuracy_centralized)


            loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
            self.loss_list_test.append(loss_test)
            self.acc_list_test.append(accuracy_test)
            accuracy_test_centralized, loss_test_centralized = self.test_latest_centralized_model_on_evaldata(self.centralized_model)
            self.loss_list_test_centralized.append(loss_test_centralized)
            self.acc_list_test_centralized.append(accuracy_test_centralized)

            selected_clients = self.select_clients(seed=round_i)

            solns, stats = self.local_train(round_i, selected_clients)

            self.metrics.extend_commu_stats(round_i, stats)

            self.latest_model = self.aggregate(solns)

            self.centralized_model = self.train_centralized(self.centralized_model)
            self.last_centralized_model = get_flat_params_from(self.centralized_model)







        out = self.get_out()
        centralized_out = self.get_centralized_out(self.centralized_model)

        weight_diff = torch.sqrt(torch.mean((self.latest_model - self.last_centralized_model) ** 2)).item()  # RMSE
        self.weight_diff_fed_centralized.append(weight_diff)

        differ = torch.sqrt(torch.mean((out - centralized_out) ** 2)).item()  # RMSE
        self.diff_fed_centralized.append(differ)

        _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        self.loss_list_train.append(loss)
        self.acc_list_train.append(accuracy)
        accuracy_centralized, loss_centralized = self.test_latest_centralized_model_on_traindata(self.centralized_model)
        self.loss_list_train_centralized.append(loss_centralized)
        self.acc_list_train_centralized.append(accuracy_centralized)

        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.loss_list_test.append(loss_test)
        self.acc_list_test.append(accuracy_test)
        accuracy_test_centralized, loss_test_centralized = self.test_latest_centralized_model_on_evaldata(self.centralized_model)
        self.loss_list_test_centralized.append(loss_test_centralized)
        self.acc_list_test_centralized.append(accuracy_test_centralized)

        np.save(loss_dir + '/loss_train_centralized_width128_' + self.dataset + self.model, self.loss_list_train_centralized)
        np.save(acc_dir + '/acc_train_centralized_width128_' + self.dataset + self.model, self.acc_list_train_centralized)
        np.save(loss_dir + '/loss_test_centralized_width128_' + self.dataset + self.model, self.loss_list_test_centralized)
        np.save(acc_dir + '/acc_test_centralized_width128_' + self.dataset + self.model, self.acc_list_test_centralized)

        np.save(loss_dir + '/loss_train_width128_' + self.dataset + self.model, self.loss_list_train)
        np.save(acc_dir + '/acc_train_width128_' + self.dataset + self.model, self.acc_list_train)
        np.save(loss_dir + '/loss_test_width128_' + self.dataset + self.model, self.loss_list_test)
        np.save(acc_dir + '/acc_test_width128_' + self.dataset + self.model, self.acc_list_test)

        np.save(output_dir + '/output_width128_'+ self.model + self.dataset, self.diff_fed_centralized)
        np.save(weight_change_dir + '/weight_change_width128_' + self.model+ self.dataset, self.weight_diff_fed_centralized)

        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()

    def train_centralized(self, model):
        model.train()
        x, y = next(iter(self.centralized_train_dataloader))
        current_batch_size = x.shape[0]
        x = x.reshape(current_batch_size, -1)
        if self.gpu:
            x, y = x.cuda(), y.cuda()

        self.centralized_optimizer.zero_grad()
        pred = model(x)

        loss = nn.MSELoss()(pred, y.unsqueeze(1).float())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 60)
        self.centralized_optimizer.step()
        return model

    def get_centralized_out(self, model):
        x, y = next(iter(self.centralized_train_dataloader))
        current_batch_size = x.shape[0]
        x = x.reshape(current_batch_size, -1)
        if self.gpu:
            x, y = x.cuda(), y.cuda()

        self.centralized_optimizer.zero_grad()
        pred = model(x).squeeze()
        return pred
