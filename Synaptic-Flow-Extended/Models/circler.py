import math
import torch
import torch.nn as nn
from Layers import layers
import operator
from functools import reduce
import numpy as np

import sys
sys.path.insert(1, '../plant_lottery_tickets')
import plant
import tickets


class PlantedTickets(nn.Module):

    def __init__(self, depth, width, dataset):
        super(PlantedTickets, self).__init__()
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
        return idcs


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


        layerStack = []

        for i in np.arange(0, len(self.width) - 2):
            l = layers.Linear(self.width[i], self.width[i+1])
            l.weight.data.copy_(torch.Tensor(self.weight[i]).transpose(0,1))
            l.bias.data.copy_(torch.Tensor(self.bias[i]))
            layerStack += [l, nn.ReLU()]
        i =  len(self.width) - 2
        l = layers.Linear(self.width[i], self.width[i+1])
        l.weight.data.copy_(torch.Tensor(self.weight[i]).transpose(0,1))
        l.bias.data.copy_(torch.Tensor(self.bias[i]))
        layerStack += [l]

        # layerStack += [nn.Softmax()]
        return nn.Sequential(*layerStack)


    def forward(self, x):

        y = self.net(x)
        return y

def _plantedModel(depth, width, dataset):
    return PlantedTickets(depth, width, dataset)