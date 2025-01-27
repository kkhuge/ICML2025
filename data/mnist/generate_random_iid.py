import torch
import numpy as np
import pickle
import os
import torchvision
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

    delta = len(data) // num_split
    data_lst = [data[i:i + delta] for i in range(0, delta * num_split, delta)]
    return data_lst



def choose_two_digit(split_data_lst):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 10, replace=False).tolist()
    except:
        print(available_digit)
    return lst


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
    min_number = min([len(dig) for dig in mnist_traindata])
    for number in range(10):
        mnist_traindata[number] = mnist_traindata[number][:min_number - 1]
    split_mnist_traindata = []
    for digit in mnist_traindata:
        split_mnist_traindata.append(data_split(digit, 10))

    mnist_testdata = []
    for number in range(10):
        idx = test_mnist.target == number
        mnist_testdata.append(test_mnist.data[idx])

    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]


    for user in range(NUM_USER):
        for d in range(10):
            l = len(split_mnist_traindata[d][-1])
            train_X[user] += split_mnist_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()

            test_X[user].append(mnist_testdata[d][0].tolist())
            test_X[user].append(mnist_testdata[d][1].tolist())
            test_X[user].append(mnist_testdata[d][2].tolist())
            test_X[user].append(mnist_testdata[d][3].tolist())
            test_X[user].append(mnist_testdata[d][4].tolist())
            test_X[user].append(mnist_testdata[d][5].tolist())
            test_X[user].append(mnist_testdata[d][6].tolist())
            test_X[user].append(mnist_testdata[d][7].tolist())
            test_X[user].append(mnist_testdata[d][8].tolist())
            test_X[user].append(mnist_testdata[d][9].tolist())
            mnist_testdata[d] = mnist_testdata[d][10:]
            test_y[user] += (d * np.ones(10)).tolist()

    print('>>> Set data path for MNIST.')
    image = 1 if IMAGE_DATA else 0
    train_path = '{}/data/train/all_data_{}_random_iid.pkl'.format(cpath, image)
    test_path = '{}/data/test/all_data_{}_random_iid.pkl'.format(cpath, image)

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
