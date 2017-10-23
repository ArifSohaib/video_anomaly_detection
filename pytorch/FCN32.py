"""defines FCN32 model"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 

from torchvision import models


class fcn32s(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.feats = models.vgg16(pretrained=True).features
        self.fconn = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096,4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout()
        )

        self.score = nn.Conv2d(4096, n_classes, 1)

    def forward(self, x):
        feats = self.feats(x)
        fconn = self.fconn(feats)
        score = self.score(fconn)
        return F.upsample(score, x.size()[2:],mode='bilinear')
