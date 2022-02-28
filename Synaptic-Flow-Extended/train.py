import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

import tickets

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), torch.squeeze(target).to(device=device, dtype=torch.long)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)


def eval(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), torch.squeeze(target).to(device=device, dtype=torch.long)
            output = model(data)
            ## rescale output
            # scale = tickets.rescale_output_torch(target, output)
            # output = scale*output
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(1, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1#, accuracy3

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    test_loss, accuracy1 = eval(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss, accuracy1]]
    for epoch in tqdm(range(epochs)):
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss, accuracy1 = eval(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss, accuracy1]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss', 'top1_accuracy']
    return pd.DataFrame(rows, columns=columns)


def trainReg(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device=device, dtype=torch.float)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)


def evalReg(model, loss, dataloader, device, verbose):
    model.eval()
    total = 0
    correct1 = 0
    # correct3 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device=device, dtype=torch.float)
            output = model(data)
            ## rescale output
            scale = tickets.rescale_output_torch(target, output)
            output = scale*output
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(1, dim=1)
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    if verbose:
        print('Evaluation: Average loss: {:.4f}'.format(average_loss))
    return average_loss

def train_eval_loop_reg(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose):
    test_loss = evalReg(model, loss, test_loader, device, verbose)
    rows = [[np.nan, test_loss]]
    for epoch in tqdm(range(epochs)):
        train_loss = trainReg(model, loss, optimizer, train_loader, device, epoch, verbose)
        test_loss = evalReg(model, loss, test_loader, device, verbose)
        row = [train_loss, test_loss]
        scheduler.step()
        rows.append(row)
    columns = ['train_loss', 'test_loss']
    return pd.DataFrame(rows, columns=columns)