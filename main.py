import numpy as np
import argparse
import importlib
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

from src.utils.worker_utils import read_data_Mnist, read_data_Cifar10
from config import OPTIMIZERS, DATASETS, MODEL_PARAMS, TRAINERS


def read_options():
    parser = argparse.ArgumentParser()

    parser.add_argument('--algo',
                        help='name of trainer;',
                        type=str,
                        choices=OPTIMIZERS,
                        default='fedavg11')
    parser.add_argument('--dataset',
                        help='name of dataset;',
                        type=str,
                        default='cifar10_all_data_1_linear_regression_niid')
    parser.add_argument('--model',
                        help='name of model;',
                        type=str,
                        default='linear_regression')
    parser.add_argument('--wd',
                        help='weight decay parameter;',
                        type=float,
                        default=0)
    parser.add_argument('--gpu',
                        action='store_true',
                        default=True,
                        help='use gpu (default: True)')
    parser.add_argument('--noprint',
                        action='store_true',
                        default=False,
                        help='whether to print inner result (default: False)')
    parser.add_argument('--noaverage',
                        action='store_true',
                        default=False,
                        help='whether to only average local solutions (default: True)')
    parser.add_argument('--device',
                        help='selected CUDA device',
                        default=0,
                        type=int)
    parser.add_argument('--num_round',
                        help='number of rounds to simulate;',
                        type=int,
                        default=200)
    parser.add_argument('--eval_every',
                        help='evaluate every ____ rounds;',
                        type=int,
                        default=1)
    parser.add_argument('--clients_per_round',
                        help='number of clients trained per round;',
                        type=int,
                        default=10)
    parser.add_argument('--batch_size',
                        help='batch size when clients train on data;',
                        type=int,
                        default=64)
    parser.add_argument('--num_epoch',
                        help='number of epochs when clients train on data;',
                        type=int,
                        default=2)
    parser.add_argument('--lr',
                        help='learning rate for inner solver;',
                        type=float,
                        default=0.1/4096)
    parser.add_argument('--seed',
                        help='seed for randomness;',
                        type=int,
                        default=0)
    parser.add_argument('--loss function',
                        help='CrossEntropyLoss or MSELoss;',
                        type=str,
                        default='MSELoss')
    parser.add_argument('--psi',
                        help='required accuracy;',
                        type=str,
                        default=0.9)
    parser.add_argument('--dis',
                        help='add more information;',
                        type=str,
                        default='')
    parsed = parser.parse_args()
    options = parsed.__dict__
    options['gpu'] = options['gpu'] and torch.cuda.is_available()


    # Set seeds
    np.random.seed(1 + options['seed'])
    torch.manual_seed(12 + options['seed'])
    if options['gpu']:
        torch.cuda.manual_seed_all(123 + options['seed'])

    # read data
    idx = options['dataset'].find("_")
    if idx != -1:
        dataset_name, sub_data = options['dataset'][:idx], options['dataset'][idx+1:]
    else:
        dataset_name, sub_data = options['dataset'], None
    assert dataset_name in DATASETS, "{} not in dataset {}!".format(dataset_name, DATASETS)

    options.update(MODEL_PARAMS(dataset_name, options['model']))


    trainer_path = 'src.trainers.%s' % options['algo']
    mod = importlib.import_module(trainer_path)
    trainer_class = getattr(mod, TRAINERS[options['algo']])

    max_length = max([len(key) for key in options.keys()])
    fmt_string = '\t%' + str(max_length) + 's : %s'
    print('>>> Arguments:')
    for keyPair in sorted(options.items()):
        print(fmt_string % keyPair)

    return options, trainer_class, dataset_name, sub_data


def main():
    options, trainer_class, dataset_name, sub_data = read_options()

    train_path = os.path.join('./data', dataset_name, 'data', 'train')
    test_path = os.path.join('./data', dataset_name, 'data', 'test')

    if dataset_name == 'cifar10':
        if options['model'] == 'linear_regression':
            all_data_info = read_data_Cifar10(train_path, test_path, 0, sub_data )
        else:
            all_data_info = read_data_Cifar10(train_path, test_path, 1, sub_data)
    else:
        all_data_info = read_data_Mnist(train_path, test_path, sub_data)

    trainer = trainer_class(options, all_data_info)
    trainer.train()


if __name__ == '__main__':
    main()
