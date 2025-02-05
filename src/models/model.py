import torch.nn as nn
import torch.nn.functional as F
import importlib
import numpy as np

class Linear_Regression(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(Linear_Regression, self).__init__()
        k = 128
        self.fc1 = nn.Linear(input_shape, k)
        self.fc2 = nn.Linear(k, k)
        self.fc3 = nn.Linear(k, out_dim)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, np.sqrt(1.5 / k))
                if m.bias is not None:
                    m.bias.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Logistic(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Logistic, self).__init__()
        self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        logit = self.layer(x)
        return logit


class TwoHiddenLayerFc(nn.Module): #output=10
    def __init__(self, input_shape, out_dim):
        k = 128
        super(TwoHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, k)
        self.fc2 = nn.Linear(k, k)
        self.fc3 = nn.Linear(k, out_dim)

        # # We do not set the standard initialization in figure 1
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.weight.data.normal_(0, np.sqrt(1.5 / k))
        #         if m.bias is not None:
        #             m.bias.data.normal_(0, 0.1)


    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out



class LeNet(nn.Module):
    def __init__(self, input_shape, out_dim):
        super(LeNet, self).__init__()
        k = 1
        self.conv1 = nn.Conv2d(input_shape[0], 6*k, 5)
        self.conv2 = nn.Conv2d(6*k, 16*k, 5)
        self.fc1 = nn.Linear(16*5*5*k, 120*k)
        self.fc2 = nn.Linear(120*k, 84*k)
        self.fc3 = nn.Linear(84*k, out_dim)

        # # We do not set the standard initialization in figure 1
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, mean=0, std=np.sqrt(1 / (m.out_channels)))
        #         if m.bias is not None:
        #             nn.init.normal_(m.bias, mean=0, std=0.1)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, mean=0, std=np.sqrt(1 / m.in_features))
        #         if m.bias is not None:
        #             nn.init.normal_(m.bias, mean=0, std=0.1)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out






def choose_model(options):
    model_name = str(options['model']).lower()
    if model_name == 'linear_regression':
        return Linear_Regression(options['input_shape'], options['num_class'])
    elif model_name == '2nn':
        return TwoHiddenLayerFc(options['input_shape'], options['num_class'])
    elif model_name == 'lenet':
        return LeNet(options['input_shape'], options['num_class'])
    elif model_name.startswith('resnet'):
        mod = importlib.import_module('src.models.resnet')
        resnet_model = getattr(mod, model_name)
        return resnet_model(options['num_class'])
    else:
        raise ValueError("Not support model: {}!".format(model_name))
