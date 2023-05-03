import torch


def exclusion_range(idx: torch.Tensor, n: torch.Tensor):
    r = torch.arange(n, device=idx.device)
    return r[r != idx]


def select_not_unbatched(t: torch.Tensor, y_single: torch.Tensor):
    print(t.device)
    print(y_single.device)
    excl = exclusion_range(y_single, t.shape[0])
    print(excl.device)
    return torch.flatten(t[excl, :])


def select_not(t: torch.Tensor, y: torch.Tensor):
    # TODO: Vectorize this if it becomes a bottleneck (as of this commit it isn't).
    single_selections = [select_not_unbatched(t, y_single) for y_single in y]
    return torch.stack(single_selections, dim=0)
