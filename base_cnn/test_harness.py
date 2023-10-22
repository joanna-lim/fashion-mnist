import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# order of tuning: size of fully connected layers, batch size, choice of activation functions

# hyperparameters to tune in this function:
    # batch size: [32, 64, 128] 
def load_dataset(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def prep_pixels(train_loader, test_loader):
    if isinstance(train_loader.dataset, list):
        for data, target in train_loader.dataset:
            data = data.float() / 255.0
    else:
        train_loader.dataset.data = train_loader.dataset.data.float() / 255.0

    if isinstance(test_loader.dataset, list):
        for data, target in test_loader.dataset:
            data = data.float() / 255.0
    else:
        test_loader.dataset.data = test_loader.dataset.data.float() / 255.0

    return train_loader, test_loader


# hyperparameters to tune in this function:
    # number of layers (if have time): [2, 3, 4, 5]
    # size of fully connected layers: [128, 256, 512]
    # choice of activation function: [relu, leaky relu, elu, prelu]

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

from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
from copy import deepcopy

def train_and_evaluate_model_kfold(model, train_dataset, test_dataset, batch_size):
    criterion = nn.CrossEntropyLoss()

    combined_dataset = ConcatDataset([train_dataset, test_dataset])
    combined_data_list = list(combined_dataset)
    kf = KFold(n_splits=5, shuffle=True)
    accuracies = []
    fold = 0
    for train_index, test_index in kf.split(combined_data_list):
        fold += 1
        curr_model = deepcopy(model)
        optimizer = optim.SGD(curr_model.parameters(), lr=0.01, momentum=0.9)
        train_data = [combined_data_list[i] for i in train_index]
        test_data = [combined_data_list[i] for i in test_index]

        train_loader, test_loader = load_dataset(train_data, test_data, batch_size)
        train_loader, test_loader = prep_pixels(train_loader, test_loader)

        best_accuracy = 0.0
        patience_counter = 0

        total_step = len(train_loader)
        for epoch in range(20):
            pbar = tqdm(enumerate(train_loader), total=total_step)
            for i, (images, labels) in pbar:
                optimizer.zero_grad()
                outputs = curr_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if (i + 1) % 200 == 0:
                    pbar.set_description(f'Fold {fold}, Epoch [{epoch + 1}/{20}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')

            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                pbar_test = tqdm(test_loader)
                for images, labels in pbar_test:
                    outputs = curr_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar_test.set_description(f'Accuracy: {100 * correct / total:.2f}%')

                accuracy = 100 * correct / total

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    patience_counter = 0

                else:
                    patience_counter += 1
                    if patience_counter >= 3: # patience = 3
                        print(f'Early stopping at epoch {epoch + 1} for fold {fold}')
                        break

        accuracies.append(best_accuracy)

    avg_accuracy = np.mean(accuracies)
    return avg_accuracy


def run_test_harness(train_dataset, test_dataset, batch_size, fc_layer_size, activation_fn):
    # train_loader, test_loader = load_dataset(train_dataset, test_dataset, batch_size)
    # train_loader, test_loader = prep_pixels(train_loader, test_loader)
    model = define_model(fc_layer_size, activation_fn)
    accuracy = train_and_evaluate_model_kfold(model, train_dataset, test_dataset, batch_size)
    return accuracy

