from src.trainers.base import BaseTrainer
from src.models.worker import LrdWorker
from torch.optim import SGD
import numpy as np
import torch
import os
from src.utils.torch_utils import set_seed
from src.models.model import TwoHiddenLayerFc

criterion = torch.nn.CrossEntropyLoss()
loss_dir = "result_loss/fedavg10"
acc_dir = "result_acc/fedavg10"


if not os.path.exists(acc_dir):
    os.makedirs(acc_dir)
if not os.path.exists(loss_dir):
    os.makedirs(loss_dir)

class FedAvg10Trainer(BaseTrainer):
    def __init__(self, options, dataset):
        self.loss_list_train = []
        self.loss_list_train_centralized = []
        self.acc_list_train = []
        self.acc_list_train_centralized = []
        self.loss_list_test = []
        self.loss_list_test_centralized = []
        self.acc_list_test = []
        self.acc_list_test_centralized = []
        self.centralized_train_dataloader_sgd = None
        seed = 12345
        set_seed(seed)
        model = TwoHiddenLayerFc(3*32*32, 10)
        self.move_model_to_gpu(model, options)
        seed = 12345
        set_seed(seed)
        self.centralized_model = TwoHiddenLayerFc(3*32*32, 10)
        self.move_model_to_gpu(self.centralized_model, options)

        self.required_accuracy = options['psi']
        self.tau = options['num_epoch']
        self.optimizer = SGD(model.parameters(), lr=options['lr'], weight_decay=0.0005)
        self.centralized_optimizer = SGD(self.centralized_model.parameters(), lr=options['lr'], weight_decay=0.0005)
        self.num_epoch = options['num_epoch']
        self.dataset = options["dataset"]
        self.loss_function = options["loss function"]
        self.model = options['model']
        worker = LrdWorker(model, self.optimizer, options)
        super(FedAvg10Trainer, self).__init__(options, dataset, worker=worker)

    def train(self):
        print('>>> Select {} clients per round \n'.format(self.clients_per_round))
        # Fetch latest flat model parameter
        self.latest_model = self.worker.get_flat_model_params().detach()
        self.last_centralized_model = self.latest_model.clone()
        for round_i in range(self.num_round):

            if round_i%1 == 0:
                _, accuracy, loss = self.test_latest_model_on_traindata(round_i)
                self.loss_list_train.append(loss)
                self.acc_list_train.append(accuracy)
                accuracy_centralized, loss_centralized = self.test_latest_centralized_model_on_traindata_sgd(self.centralized_model)
                self.loss_list_train_centralized.append(loss_centralized)
                self.acc_list_train_centralized.append(accuracy_centralized)

                loss_test, accuracy_test = self.test_latest_model_on_evaldata(round_i)
                self.loss_list_test.append(loss_test)
                self.acc_list_test.append(accuracy_test)
                accuracy_test_centralized, loss_test_centralized = self.test_latest_centralized_model_on_evaldata_sgd(self.centralized_model)
                self.loss_list_test_centralized.append(loss_test_centralized)
                self.acc_list_test_centralized.append(accuracy_test_centralized)

            selected_clients = self.select_clients(seed=round_i)
            solns, stats, data = self.local_train_dataset(round_i, selected_clients)
            self.metrics.extend_commu_stats(round_i, stats)
            self.latest_model = self.aggregate(solns)
            features, labels = next(iter(self.centralized_train_dataloader))
            self.centralized_model = self.train_centralized(self.centralized_model, features, labels)

        _, accuracy, loss = self.test_latest_model_on_traindata(self.num_round)
        self.loss_list_train.append(loss)
        self.acc_list_train.append(accuracy)
        accuracy_centralized, loss_centralized = self.test_latest_centralized_model_on_traindata_sgd(self.centralized_model)
        self.loss_list_train_centralized.append(loss_centralized)
        self.acc_list_train_centralized.append(accuracy_centralized)

        loss_test, accuracy_test = self.test_latest_model_on_evaldata(self.num_round)
        self.loss_list_test.append(loss_test)
        self.acc_list_test.append(accuracy_test)
        accuracy_test_centralized, loss_test_centralized = self.test_latest_centralized_model_on_evaldata_sgd(self.centralized_model)
        self.loss_list_test_centralized.append(loss_test_centralized)
        self.acc_list_test_centralized.append(accuracy_test_centralized)
        #width = 4096
        np.save(loss_dir + '/loss_train_centralized_width128_' + self.dataset + self.model,self.loss_list_train_centralized)
        np.save(acc_dir + '/acc_train_centralized_width128_' + self.dataset + self.model,self.acc_list_train_centralized)
        np.save(loss_dir + '/loss_test_centralized_width128_' + self.dataset + self.model,self.loss_list_test_centralized)
        np.save(acc_dir + '/acc_test_centralized_width128_' + self.dataset + self.model,self.acc_list_test_centralized)

        np.save(loss_dir + '/loss_train_width128_' + self.dataset + self.model, self.loss_list_train)
        np.save(acc_dir + '/acc_train_width128_' + self.dataset + self.model, self.acc_list_train)
        np.save(loss_dir + '/loss_test_width128_' + self.dataset + self.model, self.loss_list_test)
        np.save(acc_dir + '/acc_test_width128_' + self.dataset + self.model, self.acc_list_test)
        self.metrics.write()

    def aggregate(self, solns):
        averaged_solution = torch.zeros_like(self.latest_model)
        accum_sample_num = 0
        for num_sample, local_solution in solns:
            accum_sample_num += num_sample
            averaged_solution += num_sample * local_solution
        averaged_solution /= accum_sample_num
        return averaged_solution.detach()

    def train_centralized(self, model, feature, label):
        model.train()
        x = feature
        y = label
        current_batch_size = x.shape[0]
        x = x.reshape(current_batch_size, -1)
        if self.gpu:
            x, y = x.cuda(), y.cuda()

        self.centralized_optimizer.zero_grad()
        pred = model(x)

        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 60)
        self.centralized_optimizer.step()
        return model



    def test_latest_centralized_model_on_traindata_sgd(self, model):
        model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in self.centralized_train_dataloader:
                # from IPython import embed
                # embed()
                current_batch_size = x.shape[0]
                x = x.reshape(current_batch_size, -1)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = model(x)
                loss = criterion(pred, y)
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                test_acc += correct
            test_loss = test_loss / test_total
            test_acc = test_acc / test_total

        return test_acc , test_loss

    def test_latest_centralized_model_on_evaldata_sgd(self, model):
        model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in self.centralized_test_dataloader:
                current_batch_size = x.shape[0]
                x = x.reshape(current_batch_size, -1)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = model(x)
                loss = criterion(pred, y)
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                test_acc += correct
            test_loss = test_loss / test_total
            test_acc = test_acc / test_total

        return test_acc, test_loss




