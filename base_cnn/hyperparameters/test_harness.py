import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

# order of tuning: learning rate, size of fully connected layers, batch size, choice of activation function, early stopping

# hyperparameters to tune in this function:
    # batch size: [32, 64, 128] 
def load_dataset(train_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def prep_pixels(train_loader, test_loader):
    train_loader.dataset.data = train_loader.dataset.data.float()
    test_loader.dataset.data = test_loader.dataset.data.float()

    train_loader.dataset.data = train_loader.dataset.data / 255.0
    test_loader.dataset.data = test_loader.dataset.data / 255.0

    return train_loader, test_loader

# hyperparameters to tune in this function:
    # number of layers (if have time): [2, 3, 4, 5]
    # size of fully connected layers: [128, 256, 512]
    # choice of activation function: [relu, leaky relu, elu, prelu]
def define_model(fc_layer_size, activation_fn):
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                activation_fn, # nn.ReLU(), nn.LeakyReLU(), nn.ELU() or nn.PReLU()
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.fc1 = nn.Linear(32 * 14 * 14, fc_layer_size) # 128, 256, 512
            self.fc2 = nn.Linear(fc_layer_size, 10)

        def forward(self, x):
            out = self.layer1(x)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out

    model = CNNModel()
    return model

# hyperparameters to tune in this function:
    # learning rate: [0.001, 0.01, 0.1]
    # early stopping: [3, 4, 5, 6, 7, 8, 9, 10]
def train_and_evaluate_model(model, train_loader, test_loader, early_stopping, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_accuracy = 0.0
    best_model_weights = model.state_dict()
    patience_counter = 0

    total_step = len(train_loader)
    for epoch in range(20):
        pbar = tqdm(enumerate(train_loader), total=total_step)
        for i, (images, labels) in pbar:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 200 == 0:
                pbar.set_description(f'Epoch [{epoch + 1}/{20}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            pbar_test = tqdm(test_loader)
            for images, labels in pbar_test:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar_test.set_description(f'Accuracy: {100 * correct / total:.2f}%')
            
            accuracy = 100 * correct / total

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_weights = model.state_dict()
                patience_counter = 0
            
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f'Early stopping at epoch {epoch + 1}')
                    break

    model.load_state_dict(best_model_weights)
    return best_accuracy


def run_test_harness(train_dataset, test_dataset, batch_size, fc_layer_size, activation_fn, early_stopping, learning_rate):
    train_loader, test_loader = load_dataset(train_dataset, test_dataset, batch_size)
    train_loader, test_loader = prep_pixels(train_loader, test_loader)
    model = define_model(fc_layer_size, activation_fn)
    accuracy = train_and_evaluate_model(model, train_loader, test_loader, early_stopping, learning_rate)
    return accuracy