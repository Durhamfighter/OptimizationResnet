import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, datasets

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

def get_train_data(args):

    transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224
    transforms.ToTensor(),   # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=transform, download=True)

    return train_dataset,test_dataset

def extract_target_modules(model):
    """
    Extract specific layers (Conv2d, BatchNorm2d, Linear) from a model, excluding 'downsample' layers.

    """
    idx2name_module = {}
    i = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and 'downsample' not in name:
            idx2name_module[i] = (name, module)
            i += 1
        elif isinstance(module, nn.BatchNorm2d) and 'downsample' not in name:
            idx2name_module[i] = (name, module)
            i += 1
        elif isinstance(module, nn.Linear):
            idx2name_module[i] = (name, module)
            i += 1
    
    return idx2name_module