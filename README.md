# Widening the Network Matigates the Impact of Data Heterogeneity on FedAvg

This repository contains the codes of the paper Widening the Network Matigates the Impact of Data Heterogeneity on FedAvg

Our codes are based on the codes for the paper > [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/pdf/1907.02189.pdf)

## Genarating the IID and non-IID data

```
cd data/mnist
```
1. Running the `generate_mnist_iid.py` to obtain IID MNIST data while running `generate_dirichlet_distribution_niid.py` to obtain non-IID MNIST data.
2. Running the `generate_linear_regression_iid.py` to obtain IID mini-MNIST data while running `generate_linear_regression_niid.py` to obtain non-IID mini-MNIST data.
```
cd data/cifar10
```
3. Running the `generate_cifar_iid.py` to obtain IID CIFAR-10 data while running `generate_dirichlet_distribution_niid.py` to obtain non-IID CIFAR-10 data.
4. Running the `generate_linear_regression_iid.py` to obtain IID mini-CIFAR-10 data while running `generate_linear_regression_niid.py` to obtain non-IID mini-CIFAR-10 data.

## Note
In our experiment, the number of clients is M=10, and all clients participated in each aggregation process.

If the next experiments using the SGD, you should set

```
cd src/models/client.py
self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True) 
self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
```

If the next experiments using the GD, you should set

```
cd src/models/client.py
self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=Flase) 
self.test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=Flase)
```

Before running the code, it is necessary to manually set and save the results, for example：running the `main.py` using the `fedavg5` trainer with different networks to obtain figure 1, you must modify the name of the saved file to distinguish among different networks or widths

```
# train the family of fully connected network
np.save(loss_dir + '/loss_test' + self.dataset + self.model + '_fc1', self.loss_list_test)
np.save(acc_dir + '/acc_test' + self.dataset + self.model + '_fc1', self.acc_list_test)
```


## Impact of Non-IID Versus Network Width

Running the `main.py` using the `fedavg5` trainer with different networks to obtain figure 1 and figure 2.

Running the `main.py` using the `fedavg4` trainer with different networks to obtain the variation of global NTK and parameters in figure 3 and running the `main.py` using the `fedavg12` trainer with different networks to obtain the variation of local NTK in figure 3.

## Linear Approximation of FedAvg

Running the `main.py` using the `fedavg9` trainer with the fully-connected networks to obtain figure 4.

## FedAvg Evolves as Centralized Learning

Running the `main.py` using the `fedavg11` trainer with the fully-connected networks to obtain figure 6.


## Dependency

python = 3.8.18

pytorch = 1.9.1

CUDA = 11.1

Tensordboardx = 2.6.2.2

Numpy = 1.24.3


