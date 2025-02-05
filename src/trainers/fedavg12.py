from src.trainers.base import BaseTrainer
from src.models.model import choose_model
from src.models.worker import MSEWorker
from torch.optim import SGD
import numpy as np
import torch.nn as nn
import torch
import os



theta_dir = "result_theta/fedavg12"


if not os.path.exists(theta_dir):
    os.makedirs(theta_dir)


class FedAvg12Trainer(BaseTrainer):
    """
    Original Scheme
    """
    def __init__(self, options, dataset):
        self.theta_0_dic = []
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
        super(FedAvg12Trainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))

        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        paraments_0 = self.latest_model.clone()
        jacobian_0, out_0, _, _ = self.get_items(0)
        (_, jacobian_0_clinet_0), _= self.clients[0].solve_jacobian()
        theta_0_0 = torch.mm(jacobian_0_clinet_0, jacobian_0_clinet_0.T)
        for round_i in range(self.num_round):


            # Choose K clients prop to data size
            selected_clients = self.select_clients(seed=round_i)

            solns, stats, theta_0 = self.local_train_client_0(round_i, selected_clients)
            for k in range(len(theta_0)):
                diff = (torch.norm(theta_0[k] - theta_0_0, p='fro') / torch.norm(theta_0_0, p='fro')).item()
                self.theta_0_dic.append(diff)



            # Track communication cost
            self.metrics.extend_commu_stats(round_i, stats)

            # Update latest model
            self.latest_model = self.aggregate(solns)




        #width = 4096
        np.save(theta_dir + '/client_0_width_128' + self.model + self.dataset, self.theta_0_dic)

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


