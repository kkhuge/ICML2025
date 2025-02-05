from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import MSEWorker
from torch.optim import SGD
import numpy as np
import torch.nn as nn
import torch
import os



output_dir = "result_output_differ/fedavg9"
loss_dir = "result_loss/fedavg9"
acc_dir = "result_acc/fedavg9"


if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


class FedAvg9Trainer(BaseTrainer):
    """
    Original Scheme
    """
    def __init__(self, options, dataset):
        self.output_client_0_dic = []
        self.loss_client_0_dic = []
        self.parameters_client_0_dic = []
        self.output_client_0_0 = 0

        self.loss_list_train = []
        self.linear_loss_list_train = []
        self.loss_linear_client_0 = []
        self.diff_nonlinear_linear = []
        self.diff_nonlinear_linear_client_0=[]
        model = choose_model(options)
        self.move_model_to_gpu(model, options)
        self.tau = options['num_epoch']
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0005)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = MSEWorker(model, self.optimizer,options)
        super(FedAvg9Trainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        paraments_0 = self.latest_model.clone()
        jacobian_0, out_0, theta_0, _ = self.get_items(0)
        (_, jacobian_0_clinet_0), _= self.clients[0].solve_jacobian()
        for round_i in range(self.num_round):
            out = self.get_out()
            f_lin = out_0 + torch.matmul(jacobian_0, (self.latest_model - paraments_0))
            differ = torch.sqrt(torch.mean((out - f_lin) ** 2)).item()#RMSE
            self.diff_nonlinear_linear.append(differ)

            _, accuracy, loss  = self.test_latest_model_on_traindata(round_i)
            self.loss_list_train.append(loss)
            linear_loss = self.get_linear_loss_train(f_lin)
            self.linear_loss_list_train.append(linear_loss)
            selected_clients = self.select_clients(seed=round_i)
            solns, stats, output_client_0, loss_client_0, parameters_client_0 = self.local_train_client_0(round_i, selected_clients)
            if round_i == 0:
                self.output_client_0_0 = output_client_0[0]
            for i in range(len(output_client_0)):
                f_lin_client_0 = self.output_client_0_0 + torch.matmul(jacobian_0_clinet_0, (parameters_client_0[i] - paraments_0))
                differ_client_0 = torch.sqrt(torch.mean((output_client_0[i] - f_lin_client_0) ** 2)).item()  # RMSE
                self.diff_nonlinear_linear_client_0.append(differ_client_0)
                loss = self.get_linear_loss_train_client_0(f_lin_client_0)
                self.loss_linear_client_0.append(loss)
            self.loss_client_0_dic.append(torch.vstack(loss_client_0))
            self.metrics.extend_commu_stats(round_i, stats)
            self.latest_model = self.aggregate(solns)



        out = self.get_out()


        f_lin = out_0 + torch.matmul(jacobian_0, (self.latest_model - paraments_0))
        differ = torch.sqrt(torch.mean((out - f_lin) ** 2)).item()  # RMSE
        self.diff_nonlinear_linear.append(differ)


        self.loss_client_0_dic = torch.vstack(self.loss_client_0_dic).squeeze(1)
        self.loss_client_0_dic = self.loss_client_0_dic.detach().cpu()

        _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        self.loss_list_train.append(loss)
        linear_loss = self.get_linear_loss_train(f_lin)
        self.linear_loss_list_train.append(linear_loss)

        #width = 4096
        np.save(loss_dir + '/loss_train_client_0_width4096' + self.dataset + self.model, self.loss_client_0_dic)  #
        np.save(loss_dir + '/linear_loss_train_client_0_width4096' + self.dataset + self.model, self.loss_linear_client_0)  #
        np.save(output_dir + '/width_4096_diff_client_0_nonlinear_linear' + self.model + self.dataset, self.diff_nonlinear_linear_client_0)
        np.save(loss_dir + '/loss_train_width4096' + self.dataset + self.model, self.loss_list_train) #
        np.save(loss_dir + '/linear_loss_train_width4096' + self.dataset + self.model, self.linear_loss_list_train)  #
        np.save(output_dir + '/width_4096diff_nonlinear_linear' + self.model + self.dataset, self.diff_nonlinear_linear)

        # Save tracked information
        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()

    def get_linear_loss_train(self, linear_out):
        a = self.all_train_data.labels
        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        loss = nn.MSELoss()(linear_out, a).item()
        return loss

    def get_linear_loss_train_client_0(self,linear_out):
        a = self.all_train_data_client_0.labels
        a = torch.tensor(a, dtype=torch.float32, device=self.device)
        loss = nn.MSELoss()(linear_out, a).item()
        return loss


