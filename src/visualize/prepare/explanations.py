from typing import Iterator

import torch
from torch.utils.data import DataLoader

from prototree.models import ProtoTree, LeafRationalization


@torch.no_grad()
def data_explanations(
    tree: ProtoTree,
    loader: DataLoader,
    class_names: tuple,
) -> Iterator[tuple[LeafRationalization, int, tuple]]:
    """
    Produces an iterator of
        leaf_explanation, true_label, class_names, explanation_counter
    An iterator is used to avoid OOMing on large datasets.
    """
    for x, y in loader:
        x, y = x.to(tree.device), y.to(tree.device)
        logits, node_to_probs, predicting_leaves, leaf_explanations = tree.explain(x)
        for leaf_explanation, true_label in zip(leaf_explanations, y):
            yield leaf_explanation, true_label, class_names  # TODO: Might warrant a class.
