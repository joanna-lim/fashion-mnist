import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def load_dataset(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def prep_pixels(train_loader, test_loader):
   
    for data, target in train_loader.dataset:
        data = data.float() / 255.0
   
    for data, target in test_loader.dataset:
            data = data.float() / 255.0
    
    return train_loader, test_loader

def define_model(fc_layer_size, activation_fn):
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
            self.relu = activation_fn # nn.ReLU(), nn.LeakyReLU(), nn.ELU() or nn.PReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 14 * 14, fc_layer_size)   # 128, 256, 512
            self.fc2 = nn.Linear(fc_layer_size, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.flatten(x)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    model = CNNModel()
    return model
