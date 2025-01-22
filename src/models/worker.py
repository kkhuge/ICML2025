import numpy as np
import torch.nn as nn
from src.utils.flops_counter import get_model_complexity_info
from src.utils.torch_utils import get_flat_grad, get_state_dict, get_flat_params_from, set_flat_params_to
import torch


criterion = nn.CrossEntropyLoss()


class Worker(object):
    def __init__(self, model, optimizer, options):
        # Basic parameters
        self.model = model
        self.optimizer = optimizer
        self.batch_size = options['batch_size']
        self.num_epoch = options['num_epoch']
        self.gpu = options['gpu'] if 'gpu' in options else False
        if options["model"] == '2nn' or options["model"] == "linear_regression":
            self.flat_data = True
        else:
            self.flat_data = False

        self.flops, self.params_num, self.model_bytes = \
            get_model_complexity_info(self.model, options['input_shape'], gpu=options['gpu'])

    @property
    def model_bits(self):
        return self.model_bytes * 8
    
    def flatten_data(self, x):
        if self.flat_data:
            current_batch_size = x.shape[0]
            return x.reshape(current_batch_size, -1)
        else:
            return x

    def get_model_params(self):
        state_dict = self.model.state_dict()
        return state_dict

    def set_model_params(self, model_params_dict: dict):
        state_dict = self.model.state_dict()
        for key, value in state_dict.items():
            state_dict[key] = model_params_dict[key]
        self.model.load_state_dict(state_dict)

    def load_model_params(self, file):
        model_params_dict = get_state_dict(file)
        self.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        flat_params = get_flat_params_from(self.model)
        return flat_params.detach()

    def set_flat_model_params(self, flat_params):
        set_flat_params_to(self.model, flat_params)



    def local_train(self, train_dataloader, **kwargs):

        self.model.train()
        train_loss = train_acc = train_total = 0
        for epoch in range(self.num_epoch):
            train_loss = train_acc = train_total = 0
            for batch_idx, (x, y) in enumerate(train_dataloader):

                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                self.optimizer.zero_grad()
                pred = self.model(x)

                loss = criterion(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm(self.model.parameters(), 60)
                self.optimizer.step()

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()
                target_size = y.size(0)

                train_loss += loss.item() * y.size(0)
                train_acc += correct
                train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss/train_total,
                       "acc": train_acc/train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:

                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum()

                test_acc += correct.item()
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss


class MSEWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(MSEWorker, self).__init__(model, optimizer, options)

    def local_train(self, train_dataloader, **kwargs):

        self.model.train()
        train_loss = 0.
        train_total = 0
        train_acc = 0
        for i in range(self.num_epoch):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = nn.MSELoss()(pred, y.unsqueeze(1).float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step()
            target_size = y.size(0)

            predicted = pred.squeeze(1)
            predicted = (predicted >= 0.5).int()
            correct = predicted.eq(y).sum().item()
            train_acc += correct

            train_loss += loss.item() * y.size(0)
            train_total += target_size


        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_train_client_0(self, train_dataloader, **kwargs):
        # current_step = kwargs['T']
        pre_dic = []
        loss_dic = []
        para_dic = []
        self.model.train()
        train_loss = 0.
        train_total = 0
        train_acc = 0
        for i in range(self.num_epoch):
            parameters = self.get_flat_model_params().detach()
            para_dic.append(parameters)
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = nn.MSELoss()(pred, y.unsqueeze(1).float())
            loss_dic.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)
            self.optimizer.step()
            target_size = y.size(0)

            predicted = pred.squeeze(1)
            pre_dic.append(predicted)
            predicted = (predicted >= 0.5).int()
            correct = predicted.eq(y).sum().item()
            train_acc += correct

            train_loss += loss.item() * y.size(0)
            train_total += target_size


        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict, pre_dic, loss_dic, para_dic

    def local_test(self, test_dataloader):
        self.model.eval()
        test_acc = 0.
        test_loss = 0.
        test_total = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = nn.MSELoss()(pred, y.unsqueeze(1).float())

                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

                predicted = pred.squeeze(1)
                predicted = (predicted >= 0.5).int()
                correct = predicted.eq(y).sum().item()
                test_acc += correct

        return test_acc, test_loss

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += nn.MSELoss()(pred, y.unsqueeze(1).float()) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_prediction(self, dataloader):
        self.optimizer.zero_grad()
        prediction = []
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            prediction.append(pred)
        prediction = torch.cat(prediction).squeeze(1)
        return prediction


    def get_jacobian(self, dataloader):
        self.optimizer.zero_grad()
        grad_F_norm_squared = 0
        out_grad = []
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x).squeeze()
            for i in range(len(pred)):
                one_element_grad = get_flat_grad(pred[i], self.model.parameters(), create_graph=True)
                out_grad.append(one_element_grad)
        out_grad = torch.vstack(out_grad)

        return out_grad


class LrdWorker(Worker):
    def __init__(self, model, optimizer, options):
        self.num_epoch = options['num_epoch']
        super(LrdWorker, self).__init__(model, optimizer, options)
    
    def local_train(self, train_dataloader, **kwargs):
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch):
            x, y = next(iter(train_dataloader))
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size
        
        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
            "max": local_solution.max().item(),
            "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
            "loss": train_loss/train_total,
                "acc": train_acc/train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict

    def local_train_dataset(self, train_dataloader, **kwargs):
        data_dic = []
        self.model.train()
        train_loss = train_acc = train_total = 0
        for i in range(self.num_epoch):
            x, y = next(iter(train_dataloader))
            data_dic.append(x)
            data_dic.append(y)
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()

            self.optimizer.zero_grad()
            pred = self.model(x)

            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 60)

            self.optimizer.step()

            _, predicted = torch.max(pred, 1)
            correct = predicted.eq(y).sum().item()
            target_size = y.size(0)

            train_loss += loss.item() * y.size(0)
            train_acc += correct
            train_total += target_size

        local_solution = self.get_flat_model_params()
        param_dict = {"norm": torch.norm(local_solution).item(),
                      "max": local_solution.max().item(),
                      "min": local_solution.min().item()}
        comp = self.num_epoch * train_total * self.flops
        return_dict = {"comp": comp,
                       "loss": train_loss / train_total,
                       "acc": train_acc / train_total}
        return_dict.update(param_dict)
        return local_solution, return_dict, data_dic

    def local_test(self, test_dataloader):
        self.model.eval()
        test_loss = test_acc = test_total = 0.
        with torch.no_grad():
            for x, y in test_dataloader:

                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()

                pred = self.model(x)
                loss = criterion(pred, y)
                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(y).sum().item()

                test_acc += correct
                test_loss += loss.item() * y.size(0)
                test_total += y.size(0)

        return test_acc, test_loss

    def get_flat_grads(self, dataloader):
        self.optimizer.zero_grad()
        loss, total_num = 0., 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x)
            loss += criterion(pred, y) * y.size(0)
            total_num += y.size(0)
        loss /= total_num

        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_grad(self, dataloader):
        all_x = []
        all_y = []
        self.optimizer.zero_grad()
        loss = 0
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            all_x.append(x)
            all_y.append(y)
        all_x = torch.cat(all_x, dim=0)
        all_y = torch.cat(all_y, dim=0)
        pred = self.model(all_x)
        loss = criterion(pred, all_y)
        flat_grads = get_flat_grad(loss, self.model.parameters(), create_graph=True)
        return flat_grads

    def get_jacobian(self, dataloader):
        self.optimizer.zero_grad()
        out_grad = []
        for x, y in dataloader:
            x = self.flatten_data(x)
            if self.gpu:
                x, y = x.cuda(), y.cuda()
            pred = self.model(x).squeeze()
            for i in range(len(pred)):
                one_element_grad = []
                for j in range(len(pred[i])):
                    one_out_grad_flat = get_flat_grad(pred[i][j], self.model.parameters(), create_graph=True)
                    one_element_grad.append(one_out_grad_flat)
                one_element_grad = torch.hstack(one_element_grad)
                out_grad.append(one_element_grad)
        out_grad = torch.vstack(out_grad)

        return out_grad

    def get_error(self, test_dataloader):
        error = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                x = self.flatten_data(x)
                if self.gpu:
                    x, y = x.cuda(), y.cuda()
                true_value = np.zeros((len(y), 10))
                true_value[np.arange(len(y)), y] = 1

                pred = self.model(x)
                error = error + np.linalg.norm(pred - true_value,ord='fro')
                print(pred)
                print(true_value)
        return error