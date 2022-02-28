
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


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scoresWeights = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scoresWeights, a=math.sqrt(5))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()))
        fan = nn.init._calculate_correct_fan(self.scoresWeights, 'fan_in')
        bound = math.sqrt(6.0/fan)
        nn.init.uniform_(self.scoresBias, -bound, bound)
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        #nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(torch.cat((self.scoresWeights.abs().flatten(), self.scoresBias.abs().flatten())), self.sparsity)
        w = self.weight * subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        b = self.bias * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        x = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scoresWeights = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scoresWeights, a=math.sqrt(5))
        self.scoresBias = nn.Parameter(torch.Tensor(self.bias.size()))
        fan = nn.init._calculate_correct_fan(self.scoresWeights, 'fan_in')
        bound = math.sqrt(6.0/fan)
        nn.init.uniform_(self.scoresBias, -bound, bound)
        #nn.init.kaiming_uniform_(self.scoresBias, a=math.sqrt(5))

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        #nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(torch.cat((self.scoresWeights.abs().flatten(), self.scoresBias.abs().flatten())), self.sparsity)
        w = self.weight * subnet[:self.scoresWeights.numel()].view(self.scoresWeights.size())
        b = self.bias * subnet[self.scoresWeights.numel():].view(self.scoresBias.size())
        return F.linear(x, w, b)
        return x



def conv3x3(sparsity, in_planes, out_planes, stride=1):
    return SupermaskConv(sparsity, in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, sparsity, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(sparsity, in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=False)
        self.conv2 = conv3x3(sparsity, planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                SupermaskConv(sparsity, in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes, affine=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)

        return out


class ResNet18(nn.Module):

    def __init__(self, sparsity, num_classes):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        block, num_blocks = (BasicBlock, [2,2,2,2])

        self.conv1 = conv3x3(sparsity, 3, 64)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.layer1 = self._make_layer(sparsity, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(sparsity, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(sparsity, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(sparsity, block, 512, num_blocks[3], stride=2)
        self.linear = SupermaskLinear(sparsity, 512*block.expansion, num_classes)


    def _make_layer(self, sparsity, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(sparsity, self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    # def _initialize_weights(self, initializer):
    #     for m in self.modules():
    #         if isinstance(m, (SupermaskLinear, SupermaskConv)):
    #             nn.init.kaiming_normal_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out