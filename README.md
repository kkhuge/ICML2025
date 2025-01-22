# Widening the Network Matigates the Impact of Data Heterogeneity on FedAvg

This repository contains the codes of the paper Widening the Network Matigates the Impact of Data Heterogeneity on FedAvg

Our code based on the paper > [On the Convergence of FedAvg on Non-IID Data](https://arxiv.org/pdf/1907.02189.pdf)

## Genarating the IID and non-IID data

```
cd data/mnist
```
1. Running the generate_mnist_iid.py to obtain IID MNIST data while running generate_dirichlet_distribution_niid.py to obtain non-IID MNIST data.
2. Running the generate_linear_regression_iid.py to obtain IID mini-MNIST data while running generate_linear_regression_niid.py to obtain non-IID mini-MNIST data.
```
cd data/cifar10
```
3. Running the generate_cifar_iid.py to obtain IID CIFAR-10 data while running generate_dirichlet_distribution_niid.py to obtain non-IID CIFAR-10 data.
4. Running the generate_linear_regression_iid.py to obtain IID mini-CIFAR-10 data while running generate_linear_regression_niid.py to obtain non-IID mini-CIFAR-10 data.
