import torch.optim as optim
import argparse
import torch

def get_optimizer(args,network):
    optimizer = torch.optim.SGD(network.parameters(),
                                lr=args.lr, 
                                momentum=args.momentum, 
                                weight_decay=args.weight_decay)   
    return optimizer