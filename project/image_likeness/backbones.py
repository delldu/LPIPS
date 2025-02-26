"""Image Likeness Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

import os
from collections import namedtuple
import torch
import torch.nn as nn
from torchvision import models
import pdb


class squeezenet(nn.Module):
    def __init__(self):
        super(squeezenet, self).__init__()

        checkpoint = os.path.dirname(__file__) + "/models/squeeze.backbone.pth"
        if not os.path.exists(checkpoint):
            squeeze_pretrained_features = models.squeezenet1_1(pretrained=True).features
            torch.save(squeeze_pretrained_features.state_dict(), checkpoint)
        else:
            squeeze_pretrained_features = models.squeezenet1_1(pretrained=False).features
            squeeze_pretrained_features.load_state_dict(torch.load(checkpoint))

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.slice6 = nn.Sequential()
        self.slice7 = nn.Sequential()
        self.N_slices = 7
        for x in range(2):
            self.slice1.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(10, 11):
            self.slice5.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(11, 12):
            self.slice6.add_module(str(x), squeeze_pretrained_features[x])
        for x in range(12, 13):
            self.slice7.add_module(str(x), squeeze_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple(
            "SqueezeOutputs",
            ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"],
        )
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)

        return out


class alexnet(nn.Module):
    def __init__(self):
        super(alexnet, self).__init__()

        checkpoint = os.path.dirname(__file__) + "/models/alex.backbone.pth"
        if not os.path.exists(checkpoint):
            alexnet_pretrained_features = models.alexnet(pretrained=True).features
            torch.save(alexnet_pretrained_features.state_dict(), checkpoint)
        else:
            alexnet_pretrained_features = models.alexnet(pretrained=False).features
            alexnet_pretrained_features.load_state_dict(torch.load(checkpoint))

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"])
        return alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()

        checkpoint = os.path.dirname(__file__) + "/models/vgg16.backbone.pth"
        if not os.path.exists(checkpoint):
            vgg_pretrained_features = models.vgg16(pretrained=True).features
            torch.save(vgg_pretrained_features.state_dict(), checkpoint)
        else:
            vgg_pretrained_features = models.vgg16(pretrained=False).features
            vgg_pretrained_features.load_state_dict(torch.load(checkpoint))

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
