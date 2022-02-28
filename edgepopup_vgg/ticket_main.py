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

import ticket_models
import ticket_dataloader
import ticket_training

import numpy as np

import sys
sys.path.insert(1, '../plant_lottery_tickets')
import plant
import tickets


args = None





def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch planted ticket edge-popup implementation')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
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
    parser.add_argument('--dataset', type=str, default='circle',
                        choices=['circle', 'relu', 'helix'],
                        help='dataset (default: circle)')
    parser.add_argument('--depth', type=int, default=5, help='Depth of the network.')
    parser.add_argument('--width', type=int, default=100, help='Width of the network.')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cpu"
    args.workers = 1

    train_loader, _ = ticket_dataloader.dataloader(args.dataset, args.batch_size, True, args.workers)
    test_loader, dataTest = ticket_dataloader.dataloader(args.dataset, args.test_batch_size, False, args.workers)


    ## Model
    model = ticket_models.PlantedTickets(args.depth, args.width, args.dataset, args.sparsity).to(device)


    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )

    if (args.dataset in ['relu','helix']):
        loss = nn.MSELoss()
    else:
        loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
        )

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.sparsity == -1:
        args.sparsity = model.get_sparsity_gt()

    if (args.dataset in ['relu','helix']):
        for epoch in range(1, args.epochs + 1):
            # anneal sparsity
            if (args.anneal):
                sparsity = args.sparsity**(epoch / args.epochs)
                l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
                for layer in l:
                    if isinstance(layer, ticket_models.SupermaskLinear):
                        layer.sparsity = sparsity
            
            ticket_training.trainReg(model, device, train_loader, optimizer, loss, epoch, args.log_interval)
            ticket_training.testReg(model, device, loss, test_loader)
            scheduler.step()
    else:
        for epoch in range(1, args.epochs + 1):
            # anneal sparsity
            if (args.anneal):
                sparsity = args.sparsity**(epoch / args.epochs)
                l = [module for module in model.modules() if not isinstance(module, nn.Sequential)]
                for layer in l:
                    if isinstance(layer, ticket_models.SupermaskLinear):
                        layer.sparsity = sparsity
            
            ticket_training.trainClass(model, device, train_loader, optimizer, loss, epoch, args.log_interval)
            ticket_training.testClass(model, device, loss, test_loader)
            scheduler.step()

    ## print masks and info
    if (args.dataset == 'circle'):
        ticket = model.get_nonzero_idx_gt()
        acc = 0
        for x,y in dataTest:
            y = torch.squeeze(y).to(dtype=torch.long)
            acc += (y.numpy() == np.argmax(tickets.net_eval_classification(x.numpy(), model.weight_gt, model.bias_gt, model.scale)))
        acc /= dataTest.__len__()
        print("Sparsity of ticket: " + str(model.get_sparsity_gt()))
        print("Accuracy of ticket: " + str(acc))
        # with open(args.result_dir + '/tickets_' + args.exp_suffix + '.txt', 'w') as f:
        #     print("Accuracy: " + str(acc), file=f)
        #     for key, nzs in ticket.items():
        #         print(key + ':\n', file=f)
        #         np.savetxt(f, nzs, fmt='%1.1u')
        #         print('\n', file=f)
    
    if (args.dataset in ['relu','helix']):
        ticket = model.get_nonzero_idx_gt()
        mse = 0
        loss = nn.MSELoss()
        for x,y in dataTest:
            y = torch.squeeze(y)
            mse += np.mean((y.numpy() - tickets.net_eval(x.numpy(), model.weight_gt, model.bias_gt, model.scale))**2)
        mse /= dataTest.__len__()
        print("Sparsity of ticket: " + str(model.get_sparsity_gt()))
        print("MSE of ticket: " + str(mse))
        # with open(args.result_dir + '/tickets_' + args.exp_suffix + '.txt', 'w') as f:
        #     print("MSE: " + str(mse), file=f)
        #     for key, nzs in ticket.items():
        #         print(key + ':\n', file=f)
        #         np.savetxt(f, nzs, fmt='%1.1u')
        #         print('\n', file=f)

    if args.save_model:
        torch.save(model.state_dict(), "ticket.pt")


if __name__ == '__main__':
    main()
