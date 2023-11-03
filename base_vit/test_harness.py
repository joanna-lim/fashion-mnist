import torch
import torch.nn as nn
import os
from torch import optim
from model import VisionTransformer
from utils_vit import get_loader
from tqdm import tqdm
import contextlib

device = torch.device("mps")  

class TestHarness(object):
    def __init__(self):
        self.train_loader, self.test_loader = get_loader()
        self.model = VisionTransformer().to(device)
        self.ce = nn.CrossEntropyLoss()
    
    def train_and_evaluate(self):
        optimizer = optim.AdamW(self.model.parameters(), 5e-4, weight_decay=1e-3)
        cos_decay = optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, verbose=True)
        total_step = len(self.train_loader)

        accuracies = []

        for epoch in range(200):
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
        
