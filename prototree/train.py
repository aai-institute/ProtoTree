import logging

import torch
from torch.masked import masked_tensor
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import Leaf, Node, NodeProbabilities

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

        scaling_factor = 1 - 1 / n_batches
        tree.eval()
        update_leaf_distributions(
            tree.tree_root, y, logits.detach(), node_to_prob, scaling_factor
        )

        y_pred = torch.argmax(logits, dim=1)
        acc = torch.sum(y_pred == y).item() / len(x)
        tqdm_loader.set_postfix_str(f"batch: loss={loss.item():.5f}, {acc=:.5f}")
        total_loss += loss.item()
        total_acc += acc

        if batch_num == n_batches - 1:  # TODO: Hack due to https://github.com/tqdm/tqdm/issues/1369
            avg_loss = total_loss / n_batches
            avg_acc = total_acc / n_batches
            tqdm_loader.set_postfix_str(f"average: loss={total_loss:.5f}, acc={total_acc:.5f}")

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
    scaling_factor: float,
):
    """
    :param root:
    :param y_true: shape (batch_size)
    :param logits: shape (batch_size, num_classes)
    :param node_to_prob:
    :param scaling_factor:
    """
    batch_size, num_classes = logits.shape

    # TODO: Ideally we could to do something like
    #      y_true_range = torch.arange(0, batch_size)
    #      y_true_indices = torch.stack((y_true_range, y_true))
    #      y_true_one_hot = torch.sparse_coo_tensor(y_true_indices,
    #          torch.ones_like(y_true, dtype=torch.bool), logits.shape)
    #  but PyTorch doesn't yet have sufficient support for sparse masks.
    y_true_one_hot = F.one_hot(y_true, num_classes=num_classes).to(dtype=torch.bool)

    for leaf in root.leaves:
        update_leaf(leaf, node_to_prob, logits, y_true_one_hot, scaling_factor)


def update_leaf(
    leaf: Leaf,
    node_to_prob: dict[Node, NodeProbabilities],
    logits: torch.Tensor,
    y_true_one_hot: torch.Tensor,
    scaling_factor: float,
):
    """
    :param leaf:
    :param node_to_prob:
    :param logits: of shape (batch_size, num_classes)
    :param y_true_one_hot: boolean tensor of shape (batch_size, num_classes)
    :param scaling_factor:
    :return:
    """
    # shape (batch_size, 1)
    log_p_arrival = node_to_prob[leaf].log_p_arrival.unsqueeze(1)
    # shape (num_classes). Not the same as logits, which has (batch_size, num_classes)
    leaf_logits = leaf.logits()
    masked_logits = masked_tensor(logits, y_true_one_hot)

    masked_log_combined = log_p_arrival + (leaf_logits - masked_logits)

    # TODO: Can't use logsumexp because masked tensors don't support it, will this cause problems?
    masked_combined = torch.exp(masked_log_combined)
    masked_dist_update = torch.sum(
        masked_combined,
        dim=0,
    )

    dist_update = masked_dist_update.to_tensor(0.0)

    # This scaling (subtraction of `-1/n_batches * c` in the ProtoTree paper) seems to be a form of exponentially
    # weighted moving average, designed to ensure stability of the leaf class probability distributions (
    # leaf.dist_params), by filtering out noise from minibatching in the optimization.
    leaf.dist_params.mul_(scaling_factor)

    leaf.dist_params.add_(dist_update)
