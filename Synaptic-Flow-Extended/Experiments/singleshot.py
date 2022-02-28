import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *


import sys
sys.path.insert(1, '../plant_lottery_tickets')
import plant
import datagen
import tickets
import Models.circler

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader, _ = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, noise=args.data_noise, length=args.prune_dataset_ratio * num_classes)
    train_loader, _ = load.dataloader(args.dataset, args.train_batch_size, True, args.workers, noise=args.data_noise)
    test_loader, dataTest = load.dataloader(args.dataset, args.test_batch_size, False, args.workers, noise=args.data_noise)

    ## Model
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    if (args.dataset in ['circle', 'helix', 'relu']):
        model = load.model(args.model, args.model_class)(args.depth, args.width, args.dataset).to(device)
    else:
        model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)


    ## Loss, Optimizer ##
    
    if (args.dataset in ['relu','helix']):
        loss = nn.MSELoss()
    else:
        loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)


    ## Pre-Train ##
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    if (args.dataset in ['relu','helix']):
        pre_result = train_eval_loop_reg(model, loss, optimizer, scheduler, train_loader, 
        test_loader, device, args.pre_epochs, args.verbose)
    else :
        pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
        test_loader, device, args.pre_epochs, args.verbose)

    ## Prune ##
    print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
    pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual), args.dataset)
    if args.sparsity == None:
        sparsity = 10**(-float(args.compression))
    elif args.sparsity == -1:
        sparsity = model.get_sparsity_gt()
    else:
        sparsity = args.sparsity
    prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)

    
    ## Post-Train ##
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    if (args.dataset in ['relu','helix']):
        post_result = train_eval_loop_reg(model, loss, optimizer, scheduler, train_loader, 
        test_loader, device, args.post_epochs, args.verbose)
    else :
        post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
        test_loader, device, args.post_epochs, args.verbose)

    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
    prune_result = metrics.summary(model, 
                                   pruner.scores,
                                   metrics.flop(model, input_shape, device),
                                   lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))


    # ## print masks
    # masks = pruner.get_masks()

    # # print masks to file
    # with open(args.result_dir + '/masks_' + args.exp_suffix + '.txt', 'w') as f:
    #     i = 0
    #     for mask, param in masks:
    #         if (param.dim() == 2):
    #             print('w' + str((i//2) + 1) +  ':\n', file=f)
    #             np.savetxt(f, torch.nonzero(torch.transpose(mask, 0, 1)), fmt='%1.1u')
    #             print('\n', file=f)
    #         else:
    #             print('b' + str((i//2) + 1) +  ':\n', file=f)
    #             np.savetxt(f, torch.nonzero(mask), fmt='%1.1u')
    #             print('\n', file=f)
    #         i += 1


    if type(model) is Models.circler.PlantedTickets:
        if (args.dataset == 'circle'):
            ticket = model.get_nonzero_idx_gt()
            acc = 0
            for x,y in dataTest:
                y = torch.squeeze(y).to(dtype=torch.long)
                # print(x.numpy())
                # print(y.numpy())
                # print(tickets.net_eval_classification(x.numpy(), model.weight_gt, model.bias_gt, model.scale))
                # print("----")
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
                # print(x.numpy())
                # print(y.numpy())
                # print(tickets.net_eval(x.numpy(), model.weight_gt, model.bias_gt, model.scale))
                # print("----")
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

    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))


