import torch
from torch import nn


class LoginModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(3,32,7,2, bias = False, padding=3)
    self.mp1 = nn.MaxPool2d(3, stride = 2, padding=1)
    self.conv2 = nn.Conv2d(32,64,5,bias=False, padding=2)
    self.conv3 = nn.Conv2d(64,128, 3, 2, bias=False, padding=1)
    self.conv4 = nn.Conv2d(128,256, 1,bias=False)
    self.conv5 = nn.Conv2d(256,256, 3,stride=2,bias=False, padding=1)
    self.conv6 = nn.Conv2d(256,512, 1,bias=False)
    self.head = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(), nn.Linear(512, 256, bias=False), nn.ReLU(inplace=True), nn.Linear(256, 100, bias=False))
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.mp1(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.conv4(x)
    x = self.relu(x)
    x = self.conv5(x)
    x = self.relu(x)
    x = self.conv6(x)
    x = self.relu(x)
    x = self.head(x)
    return x
