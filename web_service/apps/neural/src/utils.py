import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_model(model, name, num):
    state = torch.load(f'/home/tim/Development/python/dj/web_service/web_service/apps/neural/models/{name}{num}.data')
#     prefix = 'module.'
# #     prefix = '' 
#     n_clip = len(prefix)
#     adapted_dict = {k[n_clip:]: v for k, v in state.items()
#                     if k.startswith(prefix)}
 
    model.load_state_dict(state)
    return model

transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(240),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(240),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def get_dataloaders(train_path, test_path, batch_size):
    dataloaders = {}
    datasets = {}
    
    datasets['train'] = torchvision.datasets.ImageFolder(train_path, transform=transforms['train'])
    datasets['test'] = torchvision.datasets.ImageFolder(test_path, transform=transforms['val'])
    
    dataloaders['train'] = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    dataloaders['test'] = DataLoader(datasets['test'], batch_size=batch_size, shuffle=False)
    return datasets, dataloaders