import time
from torch.utils.data import DataLoader


class Client(object):
    def __init__(self, cid, group, train_data, test_data, batch_size, worker):
        self.cid = cid
        self.group = group
        self.worker = worker

        self.train_data = train_data
        self.test_data = test_data

        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)  #shuffle = True when GD; shuffle = Flase when GD
        self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True) #shuffle = True when GD; shuffle = Flase when GD

    def get_model_params(self):
        return self.worker.get_model_params()

    def set_model_params(self, model_params_dict):
        self.worker.set_model_params(model_params_dict)

    def get_flat_model_params(self):
        return self.worker.get_flat_model_params()

    def set_flat_model_params(self, flat_params):
        self.worker.set_flat_model_params(flat_params)

    def get_flat_grads(self):
        grad_in_tenser = self.worker.get_flat_grads(self.train_dataloader)
        return grad_in_tenser.cpu().detach().numpy()

    def get_grad(self):
        stats = {'id': self.cid}
        grad_in_tenser = self.worker.get_grad(self.train_dataloader)
        return (len(self.train_data), grad_in_tenser.cpu().detach().numpy()), stats

    def get_prediction(self):
        prediction = self.worker.get_prediction(self.train_dataloader)
        return prediction.cpu().detach().numpy()

    def get_jacobian(self):
        jacobian_in_tenser = self.worker.get_jacobian(self.train_dataloader)
        return jacobian_in_tenser

    def solve_grad(self):
        """Get model gradient with cost"""
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}
        grads = self.get_flat_grads()

        return (len(self.train_data), grads), stats

    def solve_jacobian(self):
        bytes_w = self.worker.model_bytes
        comp = self.worker.flops * len(self.train_data)
        bytes_r = self.worker.model_bytes
        stats = {'id': self.cid, 'bytes_w': bytes_w,
                 'comp': comp, 'bytes_r': bytes_r}

        jacobian = self.get_jacobian()

        return (len(self.train_data), jacobian), stats

    def local_train(self, **kwargs):

        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats = self.worker.local_train(self.train_dataloader, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats

    def local_train_dataset(self, **kwargs):
        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats, data = self.worker.local_train_dataset(self.train_dataloader, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats, data


    def local_train_client_0(self, **kwargs):
        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats, output_client_0, loss_client_0, parameters_client_0 = self.worker.local_train_client_0(self.train_dataloader, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats, output_client_0, loss_client_0, parameters_client_0

    def local_train_client_0_theta_0(self, **kwargs):
        bytes_w = self.worker.model_bytes
        begin_time = time.time()
        local_solution, worker_stats, theta_0_dic = self.worker.local_train_client_0_theta_0(self.train_dataloader, **kwargs)
        end_time = time.time()
        bytes_r = self.worker.model_bytes

        stats = {'id': self.cid, 'bytes_w': bytes_w, 'bytes_r': bytes_r,
                 "time": round(end_time-begin_time, 2)}
        stats.update(worker_stats)

        return (len(self.train_data), local_solution), stats, theta_0_dic

    def local_test(self, use_eval_data=True):
        if use_eval_data:
            dataloader, dataset = self.test_dataloader, self.test_data
        else:
            dataloader, dataset = self.train_dataloader, self.train_data

        tot_correct, loss = self.worker.local_test(dataloader)

        return tot_correct, len(dataset), loss



