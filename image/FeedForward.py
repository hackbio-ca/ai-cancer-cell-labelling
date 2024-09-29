import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
import os

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

class Net3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16*6*6, 2)
    def forward(self, img_batch):
        a = F.relu(self.conv1(img_batch))
        a = F.relu(self.conv2(a))
        a = self.pool1(a)
        a = F.relu(self.conv3(a))
        a = F.relu(self.conv4(a))
        a = self.pool2(a)
        a = F.relu(self.conv5(a))
        a = F.relu(self.conv6(a))
        a = self.pool3(a)
        a = torch.flatten(a)            # flatten all dimensions except batch
        a = self.fc(a)
        return a



def main():
    img_path = "test.png"        
    image = io.imread(img_path)
    ## Apply transformations to the image
    image = transforms.ToTensor()(image)

    # z-score normalization
    image = transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(image)

    model = Net3()
    model.load_state_dict(torch.load('trained_model.pth', map_location=torch.device('cpu'), weights_only=True))

    prediction = model(image)
    prediction = prediction.tolist()

    if prediction[0] > prediction[1]:
        print("Benign")
    else:
        print("Malignant")


if __name__ == "__main__":
    main()



