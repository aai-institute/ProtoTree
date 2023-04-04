import logging

import torch

from prototree.node import (
    InternalNode,
    Node,
    get_max_height_nodes,
    health_check,
    remove_from_tree,
)

log = logging.getLogger(__name__)


# TODO: this only prunes nodes but not the prototype layer. Thus, all prototypes are always computed
#   and maybe even projected later. This is not efficient and, importantly, wouldn't work properly for pruning
#   during training. Make an issue about this. We should be able to prune after some epochs and then continue
#   training with the pruned tree. This is not possible with the current implementation.
def prune_unconfident_leaves(root: InternalNode, leaf_pruning_threshold: float):
    def should_prune(n: Node):
        """
        True if none of the leaves of the given node predict with a confidence
        (i.e. the max of the predicted distribution) higher the given threshold.

        This means that none of the leaves have learned something useful and thus
        the entire subtree can be pruned.
        """
        for leaf in n.leaves:
            if torch.any(leaf.y_probs() > leaf_pruning_threshold):
                return False
        return True

    original_height = root.max_height()
    for node in get_max_height_nodes(root, should_prune):
        # parent cannot be root since then root would then be removed
        if node.parent.is_root:
            continue
        remove_from_tree(node)

    try:
        health_check(root, max_height=original_height)
    except AssertionError:
        log.error(
            "Pruning failed, the tree is not healthy anymore. This is probably an implementation error."
        )
        raise
