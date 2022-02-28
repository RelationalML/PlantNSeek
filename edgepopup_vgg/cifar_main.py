from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd

from resnet_new import ResNet18
from vgg_new import vgg16, vgg16_bn
import cifar_training

from initializers import *
from planting import *

import numpy as np

import sys


args = None


def load_dimension(dataset):
    if dataset == 'MNIST':
        input_shape, num_classes = (1, 28, 28), 10
    if dataset == 'CIFAR10':
        input_shape, num_classes = (3, 32, 32), 10
    if dataset == 'CIFAR100':
        input_shape, num_classes = (3, 32, 32), 100
    if dataset == 'tiny-imagenet':
        input_shape, num_classes = (3, 64, 64), 200
    if dataset == 'imagenet':
        input_shape, num_classes = (3, 224, 224), 1000
    return input_shape, num_classes

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def cosine_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            e = epoch - args.warmup_length
            es = args.epochs - args.warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * args.lr

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    def _lr_adjuster(epoch, iteration):
        lr = args.lr * (args.lr_gamma ** (epoch // args.lr_adjust))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length


def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 edge-popup implementation')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train per level (default: 10)')
    parser.add_argument('--levels', type=int, default=10, metavar='N',
                        help='number of levels to anneal (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--warmup_length', type=int, default=5)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')

    parser.add_argument('--anneal', type=bool, default=False,
                        help='whether sparsity should slowly be annealed towards target sparsity')

    parser.add_argument('--model', default='VGG16', choices=['Resnet18', 'VGG16'],
                        help='Type of model to use')
    parser.add_argument(
        "--initBias", default="kn-zero-bias", help="Bias initialization modifications",
        choices=["kn-nonzero-bias", "ortho-nonzero-bias", "kn-zero-bias", "ortho-bias-special", "ortho-zero-bias"]
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument("--dataset", default="CIFAR10", help="Data to train on.")

    parser.add_argument('--plant-model', action='store_true', default=False,
                        help='Plant based on dict specified by path-to-target.')

    parser.add_argument('--plant-path', type=str, default='./modelBest.pt',
                        help='Path to dict of model that should be planted.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    args.workers = 16

    ## Dataset
    input_shape, num_classes = load_dimension(args.dataset)
    if (args.dataset == "CIFAR10"):
        transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=args.workers)

        testset = datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size,
                                                shuffle=False, num_workers=args.workers)

        classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    ## Model
    if (args.model == "Resnet18"):
        model = ResNet18(args.sparsity, num_classes).to(device)
    elif (args.model == "VGG16"):
        model = vgg16(args.sparsity, num_classes).to(device)

    ## Initialization
    if args.initBias == "kn-nonzero-bias":
        init_with_bias(args, model)
    if args.initBias == "kn-zero-bias":
        init_zero_bias(args, model)

    ## Planting
    if args.plant_model:
        plant_target_torch_fast(model, args.plant_path)
        print("Planting succeeded.")

    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        )


    #scheduler = CosineAnnealingLR(optimizer, T_max=10)
    scheduler = cosine_lr(optimizer, args)

    for level in range(1, args.levels + 1):
        # anneal sparsity
        if (args.anneal):
            sparsity = args.sparsity**(level / args.levels)
            print ("=====Sparsity: " + str(sparsity))
            l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
            for layer in l:
                if isinstance(layer, (resnet_new.SupermaskLinear, resnet_new.SupermaskConv)):
                    layer.sparsity = sparsity

        for epoch in range(1, args.epochs + 1):
            cifar_training.train(model, scheduler, device, train_loader, optimizer, loss, epoch, args.log_interval)
            cifar_training.test(model, device, loss, test_loader)
            #scheduler.step()


    if args.save_model:
        torch.save(model.state_dict(), "ticket.pt")


if __name__ == '__main__':
    main()
