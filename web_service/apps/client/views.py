import os
import django
from django.shortcuts import render
from PIL import Image
from io import BytesIO
from django.http import HttpResponse, JsonResponse
from django.middleware.csrf import get_token
from django.views.decorators.csrf import csrf_exempt
import torch
import torchvision.transforms as transforms
import faiss
import torchvision
from torch.utils.data import DataLoader
# from nn.nets import photo_predict
import neural.nets.center_loss
import neural.src

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        image_file = request.FILES.get('photo')
        if (image_file == None):
            return render(request, 'templates/upload.html')
        result = neural.nets.center_loss.photo_predict(image_file)
        return render(request, 'templates/prediction.html', {'prediction': result})
    return render(request, 'templates/upload.html')