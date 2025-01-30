import numpy as np
import os
import torch
import torch.nn as nn
import time
from src.models.client import Client
from src.utils.worker_utils import Metrics
from src.models.worker import Worker
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

class BaseTrainer(object):
    def __init__(self, options, dataset, model=None, optimizer=None, name='', worker=None):
        if model is not None and optimizer is not None:
            self.worker = Worker(model, optimizer, options)
        elif worker is not None:
            self.worker = worker
        else:
            raise ValueError("Unable to establish a worker! Check your input parameter!")
        print('>>> Activate a worker for training')
        self.device = options["device"]
        self.gpu = options['gpu']
        self.trainer = options['algo']
        self.batch_size = options['batch_size']
        self.all_train_data_num = 0
        _,_,self.all_train_data, self.all_test_data = dataset
        self.clients = self.setup_clients(dataset)
        assert len(self.clients) > 0
        print('>>> Initialize {} clients in total'.format(len(self.clients)))

        self.num_round = options['num_round']
        self.clients_per_round = options['clients_per_round']
        self.eval_every = options['eval_every']
        self.simple_average = not options['noaverage']
        print('>>> Weigh updates by {}'.format(
            'simple average' if self.simple_average else 'sample numbers'))

       
        self.name = '_'.join([name, f'wn{self.clients_per_round}', f'tn{len(self.clients)}'])
        self.metrics = Metrics(self.clients, options, self.name)
        self.print_result = not options['noprint']
        self.latest_model = self.worker.get_flat_model_params()

        combined_features_train = []
        combined_labels_train = []
        combined_features_test = []
        combined_labels_test = []

        transform = self.all_test_data[0].transform 
        
        for i in self.all_train_data:
            dataset = self.all_train_data[i]  
            features = torch.tensor(dataset.data)  
            labels = torch.tensor(dataset.labels)  

            combined_features_train.append(features)  
            combined_labels_train.append(labels)  

        combined_features_train = torch.cat(combined_features_train, dim=0) 
        combined_labels_train = torch.cat(combined_labels_train, dim=0)  
        self.all_train_data_client_0 = self.all_train_data[0]
        self.all_train_data = CustomDataset(data=combined_features_train,labels=combined_labels_train,transform=transform)
        if options['dataset'] == 'cifar10_all_data_1_linear_regression_niid' or options['dataset'] == 'cifar10_all_data_1_linear_regression_iid':
            self.centralized_train_dataloader = DataLoader(self.all_train_data, batch_size=500, shuffle=False)
        elif options['dataset'] == 'cifar10_all_data_1_dirichlet_niid' or options['dataset'] == 'cifar10_all_data_1_random_iid':
            self.centralized_train_dataloader = DataLoader(self.all_train_data, batch_size=640, shuffle=True)
        else:
            self.centralized_train_dataloader = DataLoader(self.all_train_data, batch_size=100, shuffle=False)


        for i in self.all_test_data:
            dataset = self.all_test_data[i]  
            features = torch.tensor(dataset.data)  
            labels = torch.tensor(dataset.labels)  

            combined_features_test.append(features)  
            combined_labels_test.append(labels)  

        combined_features_test = torch.cat(combined_features_test, dim=0)  
        combined_labels_test = torch.cat(combined_labels_test, dim=0)  
        self.all_test_data = CustomDataset(data=combined_features_test, labels=combined_labels_test,transform=transform)
        if options['dataset'] == 'cifar10_all_data_1_linear_regression_niid' or options['dataset'] == 'cifar10_all_data_1_linear_regression_iid':
            self.centralized_test_dataloader = DataLoader(self.all_test_data, batch_size=100, shuffle=False)
        else:
            self.centralized_test_dataloader = DataLoader(self.all_test_data, batch_size=100, shuffle=False)

    @staticmethod
    def move_model_to_gpu(model, options):
        if 'gpu' in options and (options['gpu'] is True):
            device = 0 if 'device' not in options else options['device']
            torch.cuda.set_device(device)
            torch.backends.cudnn.enabled = True
            model.cuda()
            print('>>> Use gpu on device {}'.format(device))
        else:
            print('>>> Don not use gpu')

    def setup_clients(self, dataset):
        """Instantiates clients based on given train and test data directories

        Returns:
            all_clients: List of clients
        """
        users, groups, train_data, test_data = dataset
        if len(groups) == 0:
            groups = [None for _ in users]

        all_clients = []
        for user, group in zip(users, groups):
            if isinstance(user, str) and len(user) >= 5:
                user_id = int(user[-5:])
            else:
                user_id = int(user)
            self.all_train_data_num += len(train_data[user])
            c = Client(user_id, group, train_data[user], test_data[user], self.batch_size, self.worker)
            all_clients.append(c)
        return all_clients

    def train(self):
        """The whole training procedure

        No returns. All results all be saved.
        """
        raise NotImplementedError

    def select_clients(self, seed=1):
        # num_clients = min(self.clients_per_round, len(self.clients))
        # np.random.seed(seed)
        # return np.random.choice(self.clients, num_clients, replace=False).tolist()
        client = []
        for i in range(10):
            client.append(self.clients[i])
        return client


    def local_train(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat = c.local_train()
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)

        return solns, stats

    def local_train_dataset(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        data_dic = []
        for i, c in enumerate(selected_clients, start=1):
            # Communicate the latest model
            c.set_flat_model_params(self.latest_model)

            # Solve minimization locally
            soln, stat, data = c.local_train_dataset()
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            # Add solutions and stats
            solns.append(soln)
            stats.append(stat)
            data_dic.append(data)

        return solns, stats, data_dic

    def local_train_client_0(self, round_i, selected_clients, **kwargs):
        """Training procedure for selected local clients

        Args:
            round_i: i-th round training
            selected_clients: list of selected clients

        Returns:
            solns: local solutions, list of the tuple (num_sample, local_solution)
            stats: Dict of some statistics
        """
        solns = []  # Buffer for receiving client solutions
        stats = []  # Buffer for receiving client communication costs
        for i, c in enumerate(selected_clients):
            c.set_flat_model_params(self.latest_model)
            if i == 0:
                if self.trainer == 'fedavg9':
                    soln, stat, output_client_0, loss_client_0, parameters_client_0 = c.local_train_client_0()
                elif self.trainer == 'fedavg12':
                    soln, stat, theta_0_dic = c.local_train_client_0_theta_0()
            else:
                soln, stat = c.local_train()
            if self.print_result:
                print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
                      "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
                      "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
                       round_i, c.cid, i, self.clients_per_round,
                       stat['norm'], stat['min'], stat['max'],
                       stat['loss'], stat['acc']*100, stat['time']))
            solns.append(soln)
            stats.append(stat)
        if self.trainer == 'fedavg9':
            return solns, stats, output_client_0, loss_client_0, parameters_client_0
        else:
            return solns, stats, theta_0_dic

    def local_test(self, use_eval_data=True):
        assert self.latest_model is not None
        # self.worker.set_flat_model_params(self.latest_model)

        num_samples = []
        tot_corrects = []
        losses = []
        for i, c in enumerate(self.clients):
            tot_correct, num_sample, loss = c.local_test(use_eval_data=use_eval_data)

            tot_corrects.append(tot_correct)
            num_samples.append(num_sample)
            losses.append(loss)

        ids = [c.cid for c in self.clients]
        groups = [c.group for c in self.clients]

        stats = {'acc': sum(tot_corrects) / sum(num_samples),
                 'loss': sum(losses) / sum(num_samples),
                 'num_samples': num_samples, 'ids': ids, 'groups': groups}

        return stats


    def test_latest_model_on_traindata(self, round_i):
        begin_time = time.time()
        stats_from_train_data = self.local_test(use_eval_data=False)

        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        stats_from_train_data['gradnorm'] = np.linalg.norm(global_grads)

        difference = 0.
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference /= len(self.clients)
        stats_from_train_data['graddiff'] = difference
        end_time = time.time()

        self.metrics.update_train_stats(round_i, stats_from_train_data)
        if self.print_result:
            print('\n>>> Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f} /'
                  ' Grad Norm: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_train_data['acc'], stats_from_train_data['loss'],
                   stats_from_train_data['gradnorm'], end_time-begin_time))
            print('=' * 102 + "\n")
        return global_grads, stats_from_train_data['acc'], stats_from_train_data['loss']


    def test_latest_model_on_evaldata(self, round_i):
        # Collect stats from total eval data
        begin_time = time.time()
        stats_from_eval_data = self.local_test(use_eval_data=True)
        end_time = time.time()

        if self.print_result and round_i % self.eval_every == 0:
            print('= Test = round: {} / acc: {:.3%} / '
                  'loss: {:.4f} / Time: {:.2f}s'.format(
                   round_i, stats_from_eval_data['acc'],
                   stats_from_eval_data['loss'], end_time-begin_time))
            print('=' * 102 + "\n")

        self.metrics.update_eval_stats(round_i, stats_from_eval_data)
        return stats_from_eval_data['loss'], stats_from_eval_data['acc']

    def test_latest_centralized_model_on_traindata(self, model):
        model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in self.centralized_train_dataloader:
                current_batch_size = x.shape[0]
                x = x.reshape(current_batch_size, -1)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = model(x)
                loss = nn.MSELoss()(pred, y.unsqueeze(1).float())
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

                predicted = pred.squeeze(1)
                predicted = (predicted >= 0.5).int()
                correct = predicted.eq(y).sum().item()
                test_acc += correct
                test_loss = test_loss / len(y)
                test_acc = test_acc / len(y)

        return test_acc , test_loss

    def test_latest_centralized_model_on_evaldata(self, model):
        model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in self.centralized_test_dataloader:
                current_batch_size = x.shape[0]
                x = x.reshape(current_batch_size, -1)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = model(x)
                loss = nn.MSELoss()(pred, y.unsqueeze(1).float())
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

                predicted = pred.squeeze(1)
                predicted = (predicted >= 0.5).int()
                correct = predicted.eq(y).sum().item()
                test_acc += correct
                test_loss = test_loss / len(y)
                test_acc = test_acc / len(y)

        return test_acc, test_loss


    def get_out(self):
        out = []
        for c in self.clients:
            prediction = c.get_prediction()
            out.append(prediction)
        out = np.concatenate(out, axis=0)
        out = torch.tensor(out, device=self.device)
        return out

    def get_items(self, round_i):
        model_len = len(self.latest_model)
        global_grads = np.zeros(model_len)
        num_samples = []
        local_grads = []
        jacobian_i = []
        out = []
        theta_i = []
        for c in self.clients:
            (num, client_jacobian), stat = c.solve_jacobian()
            jacobian_i.append(client_jacobian)
        jacobian = torch.vstack(jacobian_i)
        theta = torch.mm(jacobian, jacobian.T) / 4096

        idx = 0
        for e in range(len(jacobian_i)):
            row = len(jacobian_i[e])  
            column = len(jacobian) 
            P_i = torch.zeros((row, column), device=self.device)
            for a in range(row):
                P_i[a][a + idx] = 1
            idx = idx + row
            theta_i.append((1 / 4096) * torch.matmul(torch.matmul(jacobian, jacobian_i[e].T), P_i))

        for c in self.clients:
            (num, client_grad), stat = c.solve_grad()
            prediction = c.get_prediction()
            out.append(prediction)
            local_grads.append(client_grad)
            num_samples.append(num)
            global_grads += client_grad * num
        global_grads /= np.sum(np.asarray(num_samples))
        out = np.concatenate(out, axis=0)
        out = torch.tensor(out, device=self.device)
        return jacobian, out, theta, global_grads


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, transform=None):
        super(CustomDataset, self).__init__()
        self.data = np.array(data)
        self.labels = np.array(labels).astype("int64")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        data, target = self.data[index], self.labels[index]

        if self.transform is not None:
            data = self.transform(data)

        return data, target

