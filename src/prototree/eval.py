import logging

import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.types import SamplingStrat, SingleLeafStrat

log = logging.getLogger(__name__)


@torch.no_grad()
def eval_tree(
    tree: ProtoTree,
    data_loader: DataLoader,
    sampling_strategy: SamplingStrat = "distributed",
    desc: str = "Evaluating",
) -> float:
    """

    :param tree:
    :param data_loader:
    :param sampling_strategy:
    :param desc: description for the progress bar, passed to tqdm
    :return:
    """
    tree.eval()
    tqdm_loader = tqdm(data_loader, desc=desc, ncols=0)
    leaf_depths = []
    total_acc = 0.0
    n_batches = len(tqdm_loader)

    for batch_num, (x, y) in enumerate(tqdm_loader):
        x, y = x.to(tree.device), y.to(tree.device)
        logits, _, predicting_leaves = tree.forward(
            x, sampling_strategy=sampling_strategy
        )
        y_pred = torch.argmax(logits, dim=1)
        batch_acc = (y_pred == y).sum().item() / len(y)
        tqdm_loader.set_postfix_str(f"batch: acc={batch_acc:.5f}")
        total_acc += batch_acc

        # TODO: maybe factor out
        if predicting_leaves:
            leaf_depths.extend([leaf.depth for leaf in set(predicting_leaves)])

        if (
            batch_num == n_batches - 1
        ):  # TODO: Hack due to https://github.com/tqdm/tqdm/issues/1369
            avg_acc = total_acc / n_batches
            tqdm_loader.set_postfix_str(f"average: acc={avg_acc:.5f}")

    if leaf_depths:
        leaf_depths = np.array(leaf_depths)
        log.info(
            f"\nAverage path length is {leaf_depths.mean():.3f} with std {leaf_depths.std():.3f}"
        )
        log.info(
            f"Longest path has length {leaf_depths.max()}, shortest path has length {leaf_depths.min()}"
        )
    return avg_acc


def single_leaf_eval(
    projected_pruned_tree: ProtoTree,
    test_loader: DataLoader,
    eval_name: str,
):
    test_sampling_strategies: list[SingleLeafStrat] = ["sample_max"]
    for strategy in test_sampling_strategies:
        acc = eval_tree(
            projected_pruned_tree,
            test_loader,
            sampling_strategy=strategy,
            desc=eval_name,
        )
        fidelity = eval_fidelity(projected_pruned_tree, test_loader, strategy)

        log.info(f"Accuracy of {strategy} routing: {acc:.3f}")
        log.info(f"Fidelity of {strategy} routing: {fidelity:.3f}")


@torch.no_grad()
def eval_fidelity(
    tree: ProtoTree,
    data_loader: DataLoader,
    test_sampling_strategy: SamplingStrat,
    ref_sampling_strategy: SamplingStrat = "distributed",
) -> float:
    n_batches = len(data_loader)
    tree.eval()
    avg_fidelity = 0.0
    for x, y in tqdm(data_loader, desc="Evaluating fidelity", ncols=0):
        x, y = x.to(tree.device), y.to(tree.device)

        y_pred_reference = tree.predict(x, sampling_strategy=ref_sampling_strategy)
        y_pred_test = tree.predict(x, sampling_strategy=test_sampling_strategy)
        batch_fidelity = torch.sum(y_pred_reference == y_pred_test)
        avg_fidelity += batch_fidelity / (len(y) * n_batches)

    return avg_fidelity


# TODO: use some inbuilt of torch or sklearn
def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1.0
    else:
        return correct / total
