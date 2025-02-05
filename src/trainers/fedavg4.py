from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import MSEWorker
from torch.optim import SGD
import numpy as np
import torch
import os

weight_change_dir = "result_weight_change/fedavg4"
theta_dir = "result_theta/fedavg4"

if not os.path.exists(theta_dir):
    os.makedirs(theta_dir)
if not os.path.exists(weight_change_dir):
    os.makedirs(weight_change_dir)


class FedAvg4Trainer(BaseTrainer):
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
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0005)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = MSEWorker(model, self.optimizer,options)
        super(FedAvg4Trainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        self.latest_model = self.worker.get_flat_model_params().detach()
        paraments_0 = self.latest_model.clone()
        jacobian_0, out_0, theta_0, _ = self.get_items(0)
        for round_i in range(self.num_round):
            _, out, theta, _ = self.get_items(round_i)

            weight_change = torch.sqrt(torch.mean((self.latest_model - paraments_0) ** 2)).item() #RMSE
            self.weight_change.append(weight_change)

            f_lin = out_0 + torch.matmul(jacobian_0, (self.latest_model - paraments_0))
            differ = torch.sqrt(torch.mean((out - f_lin) ** 2)).item()#RMSE
            self.diff_nonlinear_linear.append(differ)

            self.theta.append((torch.norm(theta - theta_0, p='fro') / torch.norm(theta_0, p='fro')).item())
            print("theta_change={}".format(self.theta[-1]))

            loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
            self.loss_list_test.append(loss_test)
            self.acc_list_test.append(accuracy_test)
            selected_clients = self.select_clients(seed=round_i)
            solns, stats = self.local_train(round_i, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)
            self.latest_model = self.aggregate(solns)
        _, out, theta, _ = self.get_items(self.num_round)


        weight_change = torch.sqrt(torch.mean((self.latest_model - paraments_0) ** 2)).item()
        self.weight_change.append(weight_change)

        f_lin = out_0 + torch.matmul(jacobian_0, (self.latest_model - paraments_0))
        differ = torch.sqrt(torch.mean((out - f_lin) ** 2)).item()
        self.diff_nonlinear_linear.append(differ)

        self.theta.append((torch.norm(theta - theta_0, p='fro') / torch.norm(theta_0, p='fro')).item())
        print("theta_change={}".format(self.theta[-1]))


        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.loss_list_test.append(loss_test)
        self.acc_list_test.append(accuracy_test)
        np.save(theta_dir + '/width_64' + self.model + self.dataset, self.theta)
        np.save(weight_change_dir + '/width_64' + self.model+ self.dataset, self.weight_change)
        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()
