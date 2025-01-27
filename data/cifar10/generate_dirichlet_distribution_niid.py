import numpy as np
import pickle
import os
import torchvision
import torchvision.transforms as transforms

cpath = os.path.dirname(__file__)

NUM_USER = 10
SAVE = True
DATASET_FILE = os.path.join(cpath, 'data')
IMAGE_DATA = True
np.random.seed(6)
alpha = 0.1


class ImageDataset(object):
    def __init__(self, images, labels, normalize=False):
        self.data = images
        if normalize:
            self.data = self.data.astype(np.float32) / 255.0
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)



def main():
    # Get CIFAR-10 data
    print('>>> Get CIFAR-10 data.')
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root=DATASET_FILE, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATASET_FILE, train=False, download=True, transform=transform)

    train_cifar = ImageDataset(trainset.data, trainset.targets)
    test_cifar = ImageDataset(testset.data, testset.targets)

    distribution = np.random.dirichlet([alpha] * NUM_USER, 10)

    cifar_traindata = []
    for number in range(10):
        idx = np.array(train_cifar.target) == number
        cifar_traindata.append(train_cifar.data[idx])


    cifar_testdata = []
    for number in range(10):
        idx = np.array(test_cifar.target) == number
        cifar_testdata.append(test_cifar.data[idx])
    train_X = [[] for _ in range(NUM_USER)]
    train_y = [[] for _ in range(NUM_USER)]
    test_X = [[] for _ in range(NUM_USER)]
    test_y = [[] for _ in range(NUM_USER)]

    print(">>> Data is non-i.i.d. distributed")

    for user in range(NUM_USER):
        for i in range(10):
            num_train = np.round(distribution[i][user] * (5000-10)).astype(int)  #-10 为了保留余量，防止分配不够的情况出现
            for s in range(num_train):
                train_X[user].append(cifar_traindata[i][s].tolist())
            cifar_traindata[i] = cifar_traindata[i][num_train:]
            train_y[user].extend(np.ones(num_train) * i)

            test_X[user].append(cifar_testdata[i][-1].tolist())
            test_X[user].append(cifar_testdata[i][-2].tolist())
            test_X[user].append(cifar_testdata[i][-3].tolist())
            test_X[user].append(cifar_testdata[i][-4].tolist())
            test_X[user].append(cifar_testdata[i][-5].tolist())
            test_X[user].append(cifar_testdata[i][-6].tolist())
            test_X[user].append(cifar_testdata[i][-7].tolist())
            test_X[user].append(cifar_testdata[i][-8].tolist())
            test_X[user].append(cifar_testdata[i][-9].tolist())
            test_X[user].append(cifar_testdata[i][-10].tolist())
            cifar_testdata[i] = cifar_testdata[i][:-11]
            test_y[user] += (i * np.ones(10)).tolist()

    print('>>> Set data path for CIFAR-10.')
    image = 1 if IMAGE_DATA else 0
    train_path = '{}/data/train/all_data_{}_dirichlet_niid.pkl'.format(cpath, image)
    test_path = '{}/data/test/all_data_{}_dirichlet_niid.pkl'.format(cpath, image)

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
