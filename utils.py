import torch
import torch.nn as nn
import numpy as np


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For GPU
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def sensitivity_check(model):
    sensitivity_layer={}
    for name,module in model.named_modules():
        if isinstance(module,nn.Conv2d) and 'downsample' not in name:
            L1_weight= module.weight.data.cpu().numpy()
            L1_weight=L1_weight.reshape(L1_weight.shape[0],-1)
            L1_weight=np.sort(np.sum(np.abs(L1_weight),axis=1))[::-1]
            L1_weight=L1_weight/L1_weight[0]
            #L2_weight = torch.sqrt(torch.sum(module.weight,dim=(1,2,3)))
            sensitivity_layer[name]=L1_weight
    return sensitivity_layer