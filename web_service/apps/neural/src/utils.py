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