import torch
import numpy as np


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)

    # For GPU
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False