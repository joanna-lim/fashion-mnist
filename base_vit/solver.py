import torch
import torch.nn as nn
import os
from torch import optim
from model import VisionTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from utils_vit import get_loader
from tqdm import tqdm

device = torch.device("mps")  

class Solver(object):
    def __init__(self):
        self.train_loader, self.test_loader = get_loader()

        self.model = VisionTransformer().to(device)
        self.ce = nn.CrossEntropyLoss()

    def test_dataset(self, db='test_dataset'):
        self.model.eval()

        actual = []
        pred = []

        if db.lower() == 'train_dataset':
            loader = self.train_loader
        else:
            loader = self.test_loader

        for (imgs, labels) in loader:
            imgs = imgs.to(device)

            with torch.no_grad():
                class_out = self.model(imgs)
            _, predicted = torch.max(class_out.data, 1)

            actual += labels.tolist()
            pred += predicted.tolist()

        acc = accuracy_score(y_true=actual, y_pred=pred) * 100
        cm = confusion_matrix(y_true=actual, y_pred=pred, labels=range(10))

        return acc, cm

    def test(self):
        train_acc, cm = self.test_dataset('train')
        print("Tr Acc: %.2f" % train_acc)
        print(cm)

        test_acc, cm = self.test_dataset('test')
        print("Te Acc: %.2f" % test_acc)
        print(cm)

        return train_acc, test_acc

    def train(self):
        iter_per_epoch = len(self.train_loader)

        optimizer = optim.AdamW(self.model.parameters(), 5e-4, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, verbose=True)

        for epoch in range(3):

            self.model.train()

            for i, (imgs, labels) in enumerate(self.train_loader):

                imgs, labels = imgs.to(device), labels.to(device)

                logits = self.model(imgs)
                clf_loss = self.ce(logits, labels)

                optimizer.zero_grad()
                clf_loss.backward()
                optimizer.step()

                if i % 50 == 0 or i == (iter_per_epoch - 1):
                    print('Ep: %d/%d, it: %d/%d, err: %.4f' % (epoch + 1, 200, i + 1, iter_per_epoch, clf_loss))

            test_acc, cm = self.test_dataset('test')
            print("Test acc: %0.2f" % test_acc)
            print(cm, "\n")

            cos_decay.step()

import logging 
import contextlib
logging.disable(logging.ERROR)

class SolverTwo(object):
    def __init__(self):
        self.train_loader, self.test_loader = get_loader()
        self.model = VisionTransformer().to(device)
        self.ce = nn.CrossEntropyLoss()
    
    def train_and_evaluate(self):
        
        optimizer = optim.AdamW(self.model.parameters(), 5e-4, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, verbose=True)
        total_step = len(self.train_loader)

        accuracies = []

        for epoch in range(20):
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            self.model.train()
            for i, (images, labels) in pbar:
                with contextlib.redirect_stdout(open(os.devnull, "w")):
                    optimizer.zero_grad()
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    loss = self.ce(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    if (i + 1) % 200 == 0:
                        pbar.set_description(f'Epoch [{epoch + 1}/{20}], Step [{i + 1}/{total_step}], Loss: {loss.item()}')
                    cos_decay.step()

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                pbar_test = tqdm(self.test_loader)
                for images, labels in pbar_test:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self.model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    pbar_test.set_description(f'Accuracy: {100 * correct / total:.2f}%')

                accuracy = 100 * correct / total
                accuracies.append(accuracy)
        return accuracies
        
