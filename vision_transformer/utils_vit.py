import torch
from torchvision import datasets
from torchvision import transforms
import os


def get_loader():
    train_transform = transforms.Compose([transforms.RandomCrop(28, padding=2), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.5], [0.5])])
    test_transform = transforms.Compose([transforms.Resize([28,28]), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=test_transform)
  
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                 batch_size=128,
                                                 shuffle=True,
                                                 num_workers=4,
                                                 drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=256,
                                                shuffle=False,
                                                num_workers=4,
                                                drop_last=False)
    return train_loader, test_loader