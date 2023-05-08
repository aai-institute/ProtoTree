import torch


def select_not(t: torch.Tensor, y: torch.Tensor):
    # TODO: Vectorize this if it becomes a bottleneck (as of this commit it isn't).
    """
    Returns: A tensor where for each y_i in y, the tensor has a row with a flattened version of every row in t with
    index not equal to y_i.
    """
    single_selections = [_select_not_unbatched(t, y_single) for y_single in y]
    stacked_selections = torch.stack(single_selections, dim=0)
    return torch.flatten(stacked_selections, start_dim=1)


def _select_not_unbatched(t: torch.Tensor, y_single: torch.Tensor):
    excl = _exclusion_range(y_single, t.shape[0])
    return t[excl, :]


def _exclusion_range(idx: torch.Tensor, range_size: torch.Tensor):
    r = torch.arange(range_size, device=idx.device)
    return r[r != idx]
