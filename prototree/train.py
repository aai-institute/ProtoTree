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

    train_loader = tqdm(
        train_loader,
        desc=progress_desc,
    )
    tree.train()

    total_loss = 0.0
    total_acc = 0.0
    for x, y in train_loader:
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
        train_loader.set_postfix_str(f"(batch) loss: {loss.item():.3f}, acc: {acc:.3f}")
        total_loss += loss.item()
        total_acc += acc

    total_loss /= n_batches + 1
    total_acc /= n_batches + 1

    log.info(f"Train Epoch: Loss: {total_loss:.3f}, Acc: {total_acc:.3f}")
    return {
        "loss": total_loss,
        "train_accuracy": total_acc,
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
