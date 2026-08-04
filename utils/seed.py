import random

import numpy as np
import torch


def set_random_seed(seed):
    """Set all random-number generators used by this project to one seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Prefer repeatable CUDA kernels over the fastest kernel selection.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
