import argparse
from time import sleep
import pickle
import os
import shutil
from collections import Counter
from threading import Thread
import queue
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

from neural.src.utils import load_model
from neural.src.center_loss import CenterLoss
import torchvision.transforms as transforms

alpha = 0.005

class CenterLossClassifier(nn.Module):
    def __init__(self, train_dir):
        super(CenterLossClassifier, self).__init__()
        self.num_classes = len(os.listdir(train_dir))
        
        self.center_loss = CenterLoss(self.num_classes, feat_dim=512, use_gpu=True)
        self.optimizer_centloss = torch.optim.SGD(self.center_loss.parameters(), lr=0.5)
        
        self.model = torchvision.models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Linear(in_features, 512), 
                                      nn.ReLU(), 
                                      nn.Dropout(0.4))
        self.linear = nn.Linear(512, self.num_classes)
        
        
    def __call__(self, x):
        features = self.model(x)
        return features, self.linear(features)
    
    def check_predictions(self, dataloader):
        ys = []
        pred = []
        with torch.no_grad():
            for x, y in tqdm(dataloader):
                _, output = self(x.to(device))
                pred.append(torch.argmax(output, dim=1))
                ys.extend(y)
        correct = {}
        pred = torch.cat(pred).cpu()

        for y, p in zip(ys, pred.cpu()):
            correct[y.item()] = correct.get(y.item(), np.array([0, 0])) + np.array([y == p, 1])
        return accuracy_score(ys, pred), correct
        
    def confusion_matrix(self, dataloader):
        ys = []
        pred = []
        with torch.no_grad():
            for x, y in dataloader:
                _, output = self(x.to(device))
                pred.append(torch.argmax(output, dim=1))
                ys.extend(y)
        return confusion_matrix(ys, torch.cat(pred).cpu())   
    
    def predictions_for_class(self, x):
        with torch.no_grad():
            _, output = self(x.to(device))
            return torch.sort(torch.softmax(output.cpu(), dim=1), dim=1)

class MyDataset(Dataset):
    def __init__(self, image_path, transform=None):
        self.image = image_path
        self.transform = transform
    
    def __getitem__(self, index):
        # Загрузка изображения
        img = Image.open(self.image).convert('RGB')

        # Применение преобразований, если они заданы
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return 1

def photo_predict(photo):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    dataset = MyDataset(photo, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    train_path = "/media/tim/MSSD/DataSet/test"
    model_name = 'model'
    epoch = 4
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # print("Running eval")
    model = CenterLossClassifier(train_path).to(device)
    model = load_model(model, model_name, epoch)
    model.eval()
    
    threshold_value = 0.88 # change param
    centers_index = faiss.IndexFlatIP(512)

    centers_index.add(model.center_loss.centers.detach().cpu().numpy())
    
    distance = 1
    result = 0
    
    for x in dataloader: # x - tensor of image
        features, _ = model(x.to(device))
        for xx in features:
            d = centers_index.search(xx.detach().cpu().numpy().reshape(1, -1), 1)

            print(d[0][0])
            if d[0][0] > distance:
                return "Landmark"
            else:
                return "Non landmark"
    return result
