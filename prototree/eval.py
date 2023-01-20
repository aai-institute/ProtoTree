from collections import defaultdict

import numpy as np
import torch
import torch.optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from prototree.models import ProtoTree
from prototree.types import SamplingStrategy


@torch.no_grad()
def eval_tree(
    tree: ProtoTree,
    data_loader: DataLoader,
    sampling_strategy: SamplingStrategy = "distributed",
    eval_name: str = "Eval",
) -> float:
    tree.eval()
    tqdm_loader = tqdm(data_loader, desc=eval_name, ncols=0)
    leaf_depths = []
    for x, y in tqdm_loader:
        x, y = x.to(tree.device), y.to(tree.device)
        logits, _, predicting_leaves = tree.forward(
            x, sampling_strategy=sampling_strategy
        )
        y_pred = torch.argmax(logits, dim=1)
        acc = (y_pred == y).sum().item() / len(y)
        tqdm_loader.set_postfix_str(f"Acc: {acc:.3f}")
        # TODO: maybe factor out
        if predicting_leaves:
            leaf_depths.extend([leaf.depth() for leaf in set(predicting_leaves)])

    if leaf_depths:
        leaf_depths = np.array(leaf_depths)
        print(
            f"Average path length is {leaf_depths.mean():.2f} with std {leaf_depths.std():.2f}"
        )
        print(
            f"Longest path has length {leaf_depths.max()}, shortest path has length {leaf_depths.min()}"
        )
    return acc


@torch.no_grad()
def eval_fidelity(
    tree: ProtoTree,
    data_loader: DataLoader,
    test_sampling_strategies: tuple[SamplingStrategy] = ("sample_max", "greedy"),
    ref_sampling_strategy: SamplingStrategy = "distributed",
) -> dict[SamplingStrategy, float]:
    # TODO: is this actually true? Should be n_batches, seems like the normalization is off
    n_samples = len(data_loader)
    tree.eval()
    result_dict = defaultdict(float)
    for x, y in tqdm(data_loader, desc="Evaluating fidelity", ncols=0):
        x, y = x.to(tree.device), y.to(tree.device)

        y_pred_reference = tree.predict(x, sampling_strategy=ref_sampling_strategy)
        for sampling_strategy in test_sampling_strategies:
            y_pred_test = tree.predict(x, sampling_strategy=sampling_strategy)
            batch_fidelity = torch.sum(y_pred_reference == y_pred_test)
            result_dict[sampling_strategy] += batch_fidelity / (len(y) * n_samples)

    return result_dict


#
# @torch.no_grad()
# def eval_ensemble(
#     trees: list,
#     test_loader: DataLoader,
#     device,
#     log: Log,
#     args: argparse.Namespace,
#     sampling_strategy: str = "distributed",
#     progress_prefix: str = "Eval Ensemble",
# ):
#     # Keep an info dict about the procedure
#     info = dict()
#     # Build a confusion matrix
#     cm = np.zeros((trees[0]._num_classes, trees[0]._num_classes), dtype=int)
#
#     # Show progress on progress bar
#     test_iter = tqdm(
#         enumerate(test_loader), total=len(test_loader), desc=progress_prefix, ncols=0
#     )
#
#     # Iterate through the test set
#     for i, (xs, ys) in test_iter:
#         xs, ys = xs.to(device), ys.to(device)
#         outs = []
#         for tree in trees:
#             # Make sure the model is in evaluation mode
#             tree.eval_tree()
#             tree = tree.to(device)
#             # Use the model to classify this batch of input data
#             out, _ = tree.forward(xs, sampling_strategy)
#             outs.append(out)
#             del out
#         stacked = torch.stack(outs, dim=0)
#         ys_pred = torch.argmax(torch.mean(stacked, dim=0), dim=1)
#
#         for y_pred, y_true in zip(ys_pred, ys):
#             cm[y_true][y_pred] += 1
#
#         test_iter.set_postfix_str(f"Batch [{i + 1}/{len(test_iter)}]")
#         del outs
#
#     info["confusion_matrix"] = cm
#     info["test_accuracy"] = acc_from_cm(cm)
#     log.log_message(
#         "Ensemble accuracy with %s routing: %s"
#         % (sampling_strategy, str(info["test_accuracy"]))
#     )
#     return info


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
        return 1
    else:
        return correct / total
