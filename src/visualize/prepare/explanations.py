from typing import Iterator

import torch
from torch.utils.data import DataLoader

from proto.models import TreeSection, LeafRationalization


@torch.no_grad()
def data_explanations(
    tree: TreeSection,
    loader: DataLoader,
    class_names: tuple,
) -> Iterator[tuple[LeafRationalization, str, tuple]]:
    # TODO: This doesn't give a direct reference to the original image names/orderings (particularly if we shuffle the
    #  data. We could try to plumb that through too, but we'd need to consider a possible loss in generality, e.g. if
    #  the images don't come from files.
    """
    Produces an iterator of
        leaf_explanation, true_class, class_names
    An iterator is used to avoid OOMing on large datasets.
    """
    for x, y in loader:
        x, y = x.to(tree.device), y.to(tree.device)
        logits, node_to_probs, predicting_leaves, leaf_explanations = tree.explain(x)
        for leaf_explanation, true_label in zip(leaf_explanations, y):
            true_class = class_names[true_label]
            yield leaf_explanation, true_class, class_names  # TODO: Might warrant a class, or is that overengineering?
