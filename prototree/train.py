import logging
import math

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import Leaf, Node, NodeProbabilities

from time import time

log = logging.getLogger(__name__)


def train_epoch(
    tree: ProtoTree,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    progress_desc: str = "Train Epoch",
) -> dict:
    n_batches = len(train_loader)

    tqdm_loader = tqdm(
        train_loader,
        desc=progress_desc,
    )
    tqdm_loader.update()  # Stops earlier logging appearing after tqdm starts showing progress.
    tree.train()

    total_loss = 0.0
    total_acc = 0.0
    for batch_num, (x, y) in enumerate(tqdm_loader):
        tree.train()
        optimizer.zero_grad()
        x, y = x.to(tree.device), y.to(tree.device)
        logits, node_to_prob, predicting_leaves = tree.forward(x)
        loss = F.nll_loss(logits, y)
        loss.backward()
        optimizer.step()

        smoothing_factor = 1 - 1 / n_batches
        tree.eval()
        t1 = time()
        update_leaf_distributions(
            tree.tree_root, y, logits.detach(), node_to_prob, smoothing_factor
        )
        t2 = time()
        #print(f"{t2 - t1}")

        y_pred = torch.argmax(logits, dim=1)
        acc = torch.sum(y_pred == y).item() / len(x)
        tqdm_loader.set_postfix_str(f"batch: loss={loss.item():.5f}, {acc=:.5f}")
        total_loss += loss.item()
        total_acc += acc

        if batch_num == n_batches - 1:  # TODO: Hack due to https://github.com/tqdm/tqdm/issues/1369
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            tqdm_loader.set_postfix_str(f"average: loss={avg_loss:.5f}, acc={avg_acc:.5f}")

    return {
        "loss": avg_loss,
        "train_accuracy": avg_acc,
    }


@torch.no_grad()
def update_leaf_distributions(
    root: Node,
    y_true: torch.Tensor,
    logits: torch.Tensor,
    node_to_prob: dict[Node, NodeProbabilities],
    smoothing_factor: float,
):
    """
    :param root:
    :param y_true: shape (batch_size)
    :param logits: shape (batch_size, num_classes)
    :param node_to_prob:
    :param smoothing_factor:
    """
    for leaf in root.leaves:
        update_leaf(leaf, node_to_prob, logits, y_true, smoothing_factor)


def update_leaf(
    leaf: Leaf,
    node_to_prob: dict[Node, NodeProbabilities],
    logits: torch.Tensor,
    y_true: torch.Tensor,
    smoothing_factor: float,
):
    """
    :param leaf:
    :param node_to_prob:
    :param logits: of shape (batch_size, num_classes)
    :param y_true:
    :param smoothing_factor:
    :return:
    """
    # shape (batch_size, 1)
    log_p_arrival = node_to_prob[leaf].log_p_arrival.unsqueeze(1)
    # shape (num_classes). Not the same as logits, which has (batch_size, num_classes)
    leaf_logits = leaf.logits()

    dist_update = compute_dist_update(log_p_arrival, leaf_logits, logits, y_true)

    # This scaling (subtraction of `-1/n_batches * c` in the ProtoTree paper) seems to be a form of exponentially
    # weighted moving average, designed to ensure stability of the leaf class probability distributions (
    # leaf.dist_params), by filtering out noise from minibatching in the optimization.
    leaf.dist_params.mul_(smoothing_factor)

    leaf.dist_params.add_(dist_update)


@torch.jit.script
def logsumexp(arr: list[float]):
    if not arr:
        return -float("inf")
    c = max(arr)
    subtracted_arr = [x - c for x in arr]
    exp_arr = [math.exp(x) for x in subtracted_arr]
    arr_sum = sum(exp_arr)
    return c + math.log(arr_sum)


@torch.jit.script
def compute_dist_update(
    log_p_arrival: torch.Tensor,
    leaf_logits: torch.Tensor,
    logits: torch.Tensor,
    y_true: torch.Tensor,
):
    batch_size, num_classes = logits.shape

    log_dist_update_contributors: list[list[int]] = []
    for j in range(num_classes):
        contributors_j: list[int] = []
        log_dist_update_contributors.append(contributors_j)
    for i in range(batch_size):
        j = y_true[i]
        log_dist_update_contributors[j].append(i)

    log_dist_update = torch.zeros_like(leaf_logits)
    for j in range(num_classes):
        num_i_for_j = len(log_dist_update_contributors[j])
        #contributions_j = torch.zeros((num_i_for_j,), dtype=logits.dtype)
        contributions_j: list[float] = []

        for i in log_dist_update_contributors[j]:
            x = (log_p_arrival[i] + leaf_logits[j] - logits[i, j]).item()
            contributions_j.append(x)
        log_dist_update[j] = logsumexp(contributions_j)

    return torch.exp(log_dist_update)

