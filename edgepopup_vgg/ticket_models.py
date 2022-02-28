

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

import sys
sys.path.insert(1, '../plant_lottery_tickets')
import plant
import tickets


## =========================================================================
#### Code taken and modified from https://github.com/allenai/hidden-networks
#### Original authors: Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, Mohammad Rastegari
#### Shared under Apache License 2.0
class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class GetSubnetBias(autograd.Function):
    @staticmethod
    def forward(ctx, scoresb, k):

        # Get the supermask by sorting the scores and using the top k%
        outb = scoresb.clone()
        _, idx = scoresb.flatten().sort()
        j = int((1.0 - k) * scoresb.numel())

        # flat_out and out access the same memory.
        flat_out = outb.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return outb

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskLinear(nn.Linear):
    def __init__(self, sparsity, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        self.scoresb = nn.Parameter(torch.Tensor(self.bias.size()))
        nn.init.zeros_(self.scoresb)

        self.sparsity = sparsity

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        subnetb =  GetSubnetBias.apply(self.scoresb.abs(), self.sparsity)
        b = self.bias * subnetb
        return F.linear(x, w, b)
        return x

## =========================================================================


class PlantedTickets(nn.Module):

    def __init__(self, depth, width, dataset, sparsity):
        super(PlantedTickets, self).__init__()

        self.sparsity = sparsity
        
        self.net = self.make_layers(depth, width, dataset)
        self.dataset = dataset


    # retrieve the ground truth lottery ticket
    def get_nonzero_idx_gt(self):
        idcs = {}
        for i in np.arange(0,len(self.weight_gt)):
            nzs = np.transpose(np.nonzero(self.weight_gt[i]))
            idcs['w'+str(i+1)] = nzs
        for i in np.arange(0,len(self.bias_gt)):
            nzs = np.transpose(np.nonzero(self.bias_gt[i]))
            idcs['b'+str(i+1)] = nzs
        return


    def get_sparsity_gt(self):
        sps = 0
        paranum = 0
        for i in np.arange(0,len(self.weight_gt)):
            sps += np.count_nonzero(self.weight_gt[i])
            paranum += reduce(operator.mul, self.weight_gt[i].shape, 1)
        return sps / paranum


    def make_layers(self, depth, width, dataset):

        self.width = np.ones(depth, dtype=int)*width
        
        if dataset == 'circle':
            
            self.width[0] = 2
            self.width[-1] = 4
            K = 10
            class_bound = np.array([0.2,0.5,0.7])
            weightTarget, biasTarget = tickets.lottery_sphere(depth, class_bound, K, "class")
            weightInit, biasInit = plant.init_He_scaled(self.width)
            self.weight, self.bias, self.weight_gt, self.bias_gt, self.scale = plant.plant_target(weightTarget, biasTarget, weightInit, biasInit)

        elif dataset == 'relu':

            self.width[0] = 1
            self.width[-1] = 1
            weightTarget, biasTarget = tickets.lottery_relu(depth, "reg")
            weightInit, biasInit = plant.init_He_scaled(self.width)
            self.weight, self.bias, self.weight_gt, self.bias_gt, self.scale = plant.plant_target(weightTarget, biasTarget, weightInit, biasInit)

        elif dataset == 'helix':

            self.width[0] = 1
            self.width[-1] = 3
            weightTarget, biasTarget = tickets.lottery_helix_wide(tickets.fun_ex1, tickets.fun_ex2, tickets.fun_ex3, depth, 30)
            weightInit, biasInit = plant.init_He_scaled(self.width)
            self.weight, self.bias, self.weight_gt, self.bias_gt, self.scale = plant.plant_target(weightTarget, biasTarget, weightInit, biasInit)

        else:
            raise ValueError('Dataset ' + dataset + ' not known in connection with planted ticket networks.')

        if self.sparsity == -1:
            self.sparsity = self.get_sparsity_gt()


        layerStack = []

        for i in np.arange(0, len(self.width) - 2):
            l = SupermaskLinear(self.sparsity, self.width[i], self.width[i+1])
            l.weight.data.copy_(torch.Tensor(self.weight[i]).transpose(0,1))
            l.bias.data.copy_(torch.Tensor(self.bias[i]))
            layerStack += [l, nn.ReLU()]
        i =  len(self.width) - 2
        l = SupermaskLinear(self.sparsity, self.width[i], self.width[i+1])
        l.weight.data.copy_(torch.Tensor(self.weight[i]).transpose(0,1))
        l.bias.data.copy_(torch.Tensor(self.bias[i]))
        layerStack += [l]


        layerStack += [nn.Softmax()]
        return nn.Sequential(*layerStack)


    def forward(self, x):

        y = self.net(x)
        return y