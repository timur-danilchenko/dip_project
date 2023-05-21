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

def train_model(dataloaders, device, model, criterion, optimizer, state_path, model_name, scheduler=None, num_epochs=25, continue_train=False):
    if continue_train and os.path.exists(state_path):
        with open(state_path, 'rb') as f:
            state_dict = pickle.load(f)
        # print(state_dict)
        train_loss = state_dict['loss']
        val_loss = state_dict['val_losses']
        accuracy = state_dict['accuracy']
        start = state_dict['epoch']
        model = load_model(model, model_name, start)
        start += 1
    else:
        train_loss, val_loss, accuracy = [], [], []
        start = 0
    
    for epoch in tqdm(range(start, num_epochs)):
        train_loss.append(train_step(dataloaders, device, model, criterion, optimizer).cpu())
        cur_val_loss, cur_acc = eval_step(dataloaders, device, model)
        val_loss.append(cur_val_loss.cpu())
        accuracy.append(cur_acc)
        print(f'Accuracy is {cur_acc}')
        
        with open(state_path, 'wb') as f:
            pickle.dump({
                'loss': train_loss,
                'val_losses': val_loss,
                'epoch': epoch,
                'accuracy': accuracy
            }, f)
        torch.save(model.state_dict(), os.path.join(f'models/{model_name}{epoch}.data'))
    return train_loss, val_loss
        
def train_step(dataloaders, device, model, criterion, optimizer):
    model.train()
    total_loss = []
    iteration = 0
    for x, y in dataloaders['train']:
        optimizer.zero_grad()
        model.optimizer_centloss.zero_grad()

        x, y = x.to(device), y.to(device)
        features, output = model(x)
        loss1 = criterion(output, y) 
        loss2 = model.center_loss(features, y) * alpha
        loss = loss1 + loss2
        
        total_loss.append(loss)
        loss.backward()
    
        for param in model.center_loss.parameters():
            param.grad.data *= (1./alpha)
        model.optimizer_centloss.step()
        optimizer.step()

        iteration += 1
        if iteration % 50 == 0:
            print(f'after {iteration} loss is {loss1.item()} and {loss2.item()}')

    model.check_predictions(dataloaders['train'])
    return sum(total_loss) / len(total_loss)
            
def eval_step(dataloaders, device, model):
    model.eval()
    total_loss = []
    ys = []
    pred = []
    for x, y in dataloaders['test']: 
        with torch.no_grad():
            x, y = x.to(device), y.to(device)
            features, output = model(x)
            loss = criterion(output, y) + model.center_loss(features, y) * alpha
            total_loss.append(loss)
            
            pred.append(torch.argmax(output, dim=1))
            ys.extend(y.cpu())
            
    return sum(total_loss) / len(total_loss), accuracy_score(ys, torch.cat(pred).cpu())

def check_classes(a, b):
    return datasets['train'].classes[a] == cleanTestDataset.classes[b]

def run_train():

    b = 64
    train_again = True
    train_path = "/content/drive/MyDrive/dataset/"
    test_path = train_path
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    datasets, dataloaders = get_dataloaders(train_path, test_path, b)

    model = CenterLossClassifier(train_path).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    train_loss, val_loss = train_model(dataloaders, device, model, criterion, optimizer, state_path, model_name, 
                                        num_epochs=num_epochs, continue_train=train_again)

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