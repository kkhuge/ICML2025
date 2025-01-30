DATASETS = ['sent140', 'nist', 'shakespeare',
            'mnist', 'synthetic', 'cifar10', 'fmnist']
TRAINERS = {'fedavg4': 'FedAvg4Trainer',
            'fedavg5': 'FedAvg5Trainer',
            'fedavg9': 'FedAvg9Trainer',
            'fedavg11': 'FedAvg11Trainer',
            'fedavg12': 'FedAvg12Trainer'}
OPTIMIZERS = TRAINERS.keys()


class ModelConfig(object):
    def __init__(self):
        pass

    def __call__(self, dataset, model):
        dataset = dataset.split('_')[0]
        if dataset == 'mnist' or dataset == 'nist' or dataset == 'fmnist':
            if model == 'logistic' or model == '2nn':
                return {'input_shape': 784, 'num_class': 10}
            elif model == 'linear_regression':
                return {'input_shape': 784, 'num_class': 1}
            else:
                return {'input_shape': (1, 28, 28), 'num_class': 10}
        elif dataset == 'cifar10':
            if model == '2nn':
                return {'input_shape': 3072, 'num_class': 10}
            elif model == 'linear_regression':
                return {'input_shape': 3072, 'num_class': 1}
            else:
                return {'input_shape': (3, 32, 32), 'num_class': 10}
        else:
            raise ValueError('Not support dataset {}!'.format(dataset))


MODEL_PARAMS = ModelConfig()
