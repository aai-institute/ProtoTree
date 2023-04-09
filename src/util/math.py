import numpy as np
import torch


def log1mexp(log_p: torch.Tensor) -> torch.Tensor:
    """
    Compute `log(1-p) = log(1 - exp(log_p))` in a numerically stable way. Implementation inspired by `TensorFlow
    log1mexp <https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/math/generic.py
    #L447-L471>`_ (but note that the TensorFlow function computes something slightly different).

    :param log_p:
    :return:
    """
    log_p = log_p - 1e-7
    # noinspection PyTypeChecker
    return torch.where(
        log_p < -np.log(2),
        torch.log(-torch.expm1(log_p)),
        torch.log1p(-torch.exp(log_p)),
    )
