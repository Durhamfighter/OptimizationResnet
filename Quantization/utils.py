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


def get_train_data(args):

    transform = transforms.Compose([
    transforms.Resize(224),  # Resize to 224x224
    transforms.ToTensor(),   # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])
    train_dataset = datasets.CIFAR10(root=args.data_path, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(root=args.data_path, train=False, transform=transform, download=True)

    return train_dataset,test_dataset
