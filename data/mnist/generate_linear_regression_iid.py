import torch
import numpy as np
import pickle
import os
import torchvision
import random
cpath = os.path.dirname(__file__)


NUM_USER = 10
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = not False
np.random.seed(6)


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        if isinstance(images, torch.Tensor):
            if not IMAGE_DATA:
                self.data = images.view(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize and not IMAGE_DATA:
            mu = np.mean(self.data.astype(np.float32), 0)
            sigma = np.std(self.data.astype(np.float32), 0)
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


def main():
    print('>>> Get MNIST data.')
    trainset = torchvision.datasets.MNIST(DATASET_FILE, download=True, train=True)
    testset = torchvision.datasets.MNIST(DATASET_FILE, download=True, train=False)

    train_mnist = ImageDataset(trainset.train_data, trainset.train_labels)
    test_mnist = ImageDataset(testset.test_data, testset.test_labels)

    mnist_traindata = []
    for number in range(10):
        idx = train_mnist.target == number
        mnist_traindata.append(train_mnist.data[idx])
    mnist_traindata = mnist_traindata[:2]
    min_number = min([len(dig) for dig in mnist_traindata])
    for number in range(2):
        mnist_traindata[number] = mnist_traindata[number][:min_number - 1]
    split_mnist_traindata = []
    for digit in mnist_traindata:
        split_mnist_traindata.append(data_split(digit, 1000))

    mnist_testdata = []
    for number in range(10):
        idx = test_mnist.target == number
        mnist_testdata.append(test_mnist.data[idx])
    mnist_testdata = mnist_testdata[:2]
    min_number = min([len(dig) for dig in mnist_testdata])
    for number in range(2):
        mnist_testdata[number] = mnist_testdata[number][:min_number - 1]
    split_mnist_testdata = []
    for digit in mnist_testdata:
        split_mnist_testdata.append(data_split(digit, 1000))

    data_distribution = np.array([len(v) for v in mnist_traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_mnist_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_mnist_testdata])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(NUM_USER):
        print(user, np.array([len(v) for v in split_mnist_traindata]))

        for d in range(2):
            l = len(split_mnist_traindata[d][-1])
            train_X[user] += split_mnist_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()
            l = len(split_mnist_testdata[d][-1])
            test_X[user] += split_mnist_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()
        indices = list(range(len(train_X[user])))
        random.shuffle(indices)
        train_X[user] = [train_X[user][i] for i in indices]
        train_y[user] = [train_y[user][i] for i in indices]

        indices = list(range(len(test_X[user])))
        random.shuffle(indices)
        test_X[user] = [test_X[user][i] for i in indices]
        test_y[user] = [test_y[user][i] for i in indices]

    print('>>> Set data path for MNIST.')
    image = 1 if IMAGE_DATA else 0
    train_path = '{}/data/train/all_data_{}_linear_regression_iid.pkl'.format(cpath, image)
    test_path = '{}/data/test/all_data_{}_linear_regression_iid.pkl'.format(cpath, image)

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    for i in range(NUM_USER):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    if SAVE:
        with open(train_path, 'wb') as outfile:
            pickle.dump(train_data, outfile)
        with open(test_path, 'wb') as outfile:
            pickle.dump(test_data, outfile)

        print('>>> Save data.')


if __name__ == '__main__':
    main()
