import os
import random

import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        pass
    else:
        threadpool_limits(limits=1)
    try:
        import torch
    except ImportError:
        pass
    else:
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)

