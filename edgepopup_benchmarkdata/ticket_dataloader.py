import torch
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim

import datagen


def dataloader(dataset, batch_size, train, workers, length=None):
    # Dataset
    if dataset == 'circle':
        dataset = datagen.gen_data_circle_syn(is_train=train)

    if dataset == 'relu':
        dataset = datagen.gen_data_relu_syn(is_train=train)

    if dataset == 'helix':
        dataset = datagen.gen_data_helix_syn(is_train=train)

    
    # Dataloader
    use_cuda = torch.cuda.is_available()
    kwargs = {'num_workers': workers, 'pin_memory': True} if use_cuda else {}
    shuffle = train is True
    if length is not None:
        indices = torch.randperm(len(dataset))[:length]
        dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle, 
                                             **kwargs)

    return dataloader, dataset