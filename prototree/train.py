import logging

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.node import Leaf, Node, NodeProbabilities
from util.log import Log

logger = logging.getLogger(__name__)


def train_epoch(
    tree: ProtoTree,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    progress_desc: str = "Train Epoch",
) -> dict:
    # TODO: combine this block with the leaf distribution optimization below into a single, separate function
    tree.eval()
    n_batches = float(len(train_loader))

    train_loader = tqdm(
        train_loader,
        desc=progress_desc,
    )
    tree.train()

    total_loss = 0.0
    total_acc = 0.0
    for x, y in train_loader:
        optimizer.zero_grad()
        tree.train()
        x, y = x.to(tree.device), y.to(tree.device)
        logits, node_to_prob, predicting_leaves = tree.forward(x)
        loss = F.nll_loss(logits, y)
        loss.backward()
        optimizer.step()

        # update the leaf dist_params in a derivative-free fashion
        scaling_factor = 1 - 1 / n_batches
        tree.eval()
        update_leaf_distributions(
            tree.tree_root, y, logits, node_to_prob, scaling_factor
        )

        # Count the number of correct classifications
        y_pred = torch.argmax(logits, dim=1)
        acc = torch.sum(y_pred == y).item() / len(x)

        train_loader.set_postfix_str(f"Loss: {loss.item():.3f}, Acc: {acc:.3f}")
        # Compute metrics over this batch
        total_loss += loss.item()
        total_acc += acc

    return {
        "loss": total_loss / (n_batches + 1),
        "train_accuracy": total_acc / (n_batches + 1),
    }


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
    with torch.no_grad():  # TODO: is no_grad still needed?
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
    # TODO: why is this scaling of dist_params needed?
    leaf.dist_params *= scaling_factor
    leaf.dist_params += dist_update
    # dist_params values can get slightly negative because of floating point issues, so set to zero.
    # TODO: previously this line had no effect, I reinstated it. I hope it doesn't break stuff...
    #   Unclear why we would need this at all, dist_params only go out with softmax and can be negative
    F.relu(leaf.dist_params, inplace=True)
