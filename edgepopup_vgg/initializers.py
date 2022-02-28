import torch
import torch.nn as nn
import math
import torch.optim
import resnet_new

def init_with_bias(args, model):
    gainProd = 0.06
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0.0, std=gainProd)


def init_zero_bias(args, model):
    gainProd = 0.06 #
    for m in model.modules():
        if isinstance(m,(resnet_new.SupermaskConv, resnet_new.SupermaskLinear)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            std = math.sqrt(2/fan_in)
            if args.scale_fan:
                std = std / math.sqrt(args.sparsity)
            nn.init.normal_(m.weight, mean=0.0, std=std)
            gainProd = gainProd*std
            if m.bias is not None:
                nn.init.zeros_(m.bias)
