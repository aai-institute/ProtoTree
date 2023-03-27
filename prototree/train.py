import logging

import torch
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
            tree.tree_root, y, logits, node_to_prob, scaling_factor
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
    :param scaling_factor: usually 1 - 1/n_batches. TODO: understand its role
    :return:
    """
    num_classes = logits.shape[-1]

    log_eye = torch.log(torch.eye(num_classes, device=y_true.device))
    # one_hot encoded logits, -inf everywhere, zero at one entry
    target_logits = log_eye[y_true]
    for leaf in root.leaves:
        update_leaf(leaf, node_to_prob, logits, target_logits, scaling_factor)


def update_leaf(
    leaf: Leaf,
    node_to_prob: dict[Node, NodeProbabilities],
    logits: torch.Tensor,
    target_logits: torch.Tensor,
    scaling_factor: float,
):
    """

    :param leaf:
    :param node_to_prob:
    :param logits: of shape (batch_size, k)
    :param target_logits: of shape (batch_size, k)
    :param scaling_factor:
    :return:
    """
    # shape (batch_size, 1)
    log_p_arrival = node_to_prob[leaf].log_p_arrival.unsqueeze(1)
    # shape (num_classes). Not the same as logits, which has (batch_size, num_classes)
    leaf_logits = leaf.logits()
    # TODO: clarify what is happening here. torch.log(target) seems to contain
    #   negative infinity everywhere and zero at one place.
    #   We are also summing together tensors of different shapes.
    #   There is probably a clearer and more efficient way to compute this, also check paper
    log_dist_update = torch.logsumexp(
        log_p_arrival + leaf_logits + target_logits - logits,
        dim=0,
    )
    # should have zero everywhere except one entry
    dist_update = torch.exp(log_dist_update)

    # This scaling (subtraction of `-1/n_batches * c` in the ProtoTree paper) seems to be a form of exponentially
    # weighted moving average, designed to ensure stability of the leaf class probability distributions (
    # leaf.dist_params), by filtering out noise from minibatching in the optimization.
    leaf.dist_params.mul_(scaling_factor)

    leaf.dist_params.add_(dist_update)
