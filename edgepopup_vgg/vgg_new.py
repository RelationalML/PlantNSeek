from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import operator
from functools import reduce
import numpy as np
import torch.nn.functional as F

from resnet_new import SupermaskLinear, SupermaskConv

# Based on code taken from https://github.com/facebookresearch/open_lth
class ConvModule(nn.Module):
    def __init__(self, sparsity, in_filters, out_filters):
        super(ConvModule, self).__init__()
        self.conv = SupermaskConv(sparsity, in_filters, out_filters, kernel_size=3, padding=1)

    def forward(self, x):
        return F.relu(self.conv(x))

class ConvBNModule(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ConvBNModule, self).__init__()
        self.conv =  SupermaskConv(sparsity, in_filters, out_filters, kernel_size=3, padding=1)
        self.bn = layers.BatchNorm2d(out_filters, affine=False)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class VGG(nn.Module):
    def __init__(self, sparsity, plan, conv, num_classes):
        super(VGG, self).__init__()
        layer_list = []
        filters = 3

        for spec in plan:
            if spec == 'M':
                layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layer_list.append(conv(sparsity, filters, spec))
                filters = spec

        self.layers = nn.Sequential(*layer_list)        

        self.fc = SupermaskLinear(sparsity, 512, num_classes)


    def forward(self, x):
        x = self.layers(x)
        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def _plan(num):
    if num == 11:
        plan = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 13:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512]
    elif num == 16:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
    elif num == 19:
        plan = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    else:
        raise ValueError('Unknown VGG model: {}'.format(num))
    return plan

def _vgg(sparsity, plan, conv, num_classes):
    model = VGG(sparsity, plan, conv, num_classes)
    return model

def vgg11(sparsity, num_classes):
    plan = _plan(11)
    return _vgg(sparsity, plan, ConvModule, num_classes)

def vgg11_bn(sparsity, num_classes):
    plan = _plan(11)
    return _vgg(sparsity, plan, ConvBNModule, num_classes)

def vgg13(sparsity, num_classes):
    plan = _plan(13)
    return _vgg(sparsity, plan, ConvModule, num_classes)

def vgg13_bn(sparsity, num_classes):
    plan = _plan(13)
    return _vgg(sparsity, plan, ConvBNModule, num_classes)

def vgg16(sparsity, num_classes):
    plan = _plan(16)
    return _vgg(sparsity, plan, ConvModule, num_classes)

def vgg16_bn(sparsity, num_classes):
    plan = _plan(16)
    return _vgg(sparsity, plan, ConvBNModule, num_classes)

def vgg19(sparsity, num_classes):
    plan = _plan(19)
    return _vgg(sparsity, plan, ConvModule, num_classes)

def vgg19_bn(sparsity, num_classes):
    plan = _plan(19)
    return _vgg(sparsity, plan, ConvBNModule, num_classes)
