from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F


def min_pool2d(xs, **kwargs):
    return -F.max_pool2d(-xs, **kwargs)
