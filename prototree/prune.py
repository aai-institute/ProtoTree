from copy import deepcopy

import torch

from prototree.node import InternalNode, Leaf, Node
from prototree.prototree import ProtoTree
from util.log import Log


# Collects the nodes
def nodes_to_prune_based_on_leaf_dists_threshold(
    tree: ProtoTree, threshold: float
) -> list:
    to_prune_incl_possible_children = []
    for node in tree.all_nodes:
        if has_max_prob_lower_threshold(node, threshold):
            # prune everything below incl this node
            to_prune_incl_possible_children.append(node.index)
    return to_prune_incl_possible_children


# Returns True when all the node's children have a max leaf value < threshold
def has_max_prob_lower_threshold(node: Node, threshold: float):
    if isinstance(node, InternalNode):
        for leaf in node.leaves:
            if leaf.log_probabilities:
                if torch.max(torch.exp(leaf.distribution())).item() > threshold:
                    return False
            else:
                if torch.max(leaf.distribution()).item() > threshold:
                    return False
    elif isinstance(node, Leaf):
        if node.log_probabilities:
            if torch.max(torch.exp(node.distribution())).item() > threshold:
                return False
        else:
            if torch.max(node.distribution()).item() > threshold:
                return False
    else:
        raise RuntimeError(
            "This node type should not be possible. A tree has internal_nodes and leaves."
        )
    return True


# TODO: this only prunes nodes but not the prototype layer. Thus, all prototypes are always computed
#   and maybe even projected later. This is not efficient and, importantly, wouldn't work properly for pruning
#   during training. Make an issue about this. We should be able to prune after some epochs and then continue
#   training with the pruned tree. This is not possible with the current implementation.
def prune(tree: ProtoTree, pruning_threshold_leaves: float):
    node_idxs_to_prune = nodes_to_prune_based_on_leaf_dists_threshold(
        tree, pruning_threshold_leaves
    )
    to_prune = deepcopy(node_idxs_to_prune)
    # remove children from prune_list of nodes that would already be pruned
    for node_idx in node_idxs_to_prune:
        cur_node = tree.node_by_index[node_idx]
        if isinstance(cur_node, InternalNode):
            if node_idx > 0:  # parent cannot be root since root would then be removed
                for child in cur_node.descendants:
                    if child.index in to_prune and child.index != node_idx:
                        to_prune.remove(child.index)

    # TODO: Jesus Christ...
    for node_idx in to_prune:
        node = tree.node_by_index[node_idx]
        parent = tree.node2parent[node]
        if parent.index == 0:  # parent cannot be root since root would then be removed
            continue
        if node == parent.left:
            if parent == tree.node2parent[parent].left:
                # make right child of parent the left child of parent of parent
                tree.node2parent[parent.right] = tree.node2parent[parent]
                tree.node2parent[parent].left = parent.right
            elif parent == tree.node2parent[parent].right:
                # make right child of parent the right child of parent of parent
                tree.node2parent[parent.right] = tree.node2parent[parent]
                tree.node2parent[parent].right = parent.right
            else:
                raise RuntimeError("Pruning went wrong, this should not be possible")

        elif node == parent.right:
            if parent == tree.node2parent[parent].left:
                # make left child or parent the left child of parent of parent
                tree.node2parent[parent.left] = tree.node2parent[parent]
                tree.node2parent[parent].left = parent.left
            elif parent == tree.node2parent[parent].right:
                # make left child of parent the right child of parent of parent
                tree.node2parent[parent.left] = tree.node2parent[parent]
                tree.node2parent[parent].right = parent.left
            else:
                raise RuntimeError(
                    "Pruning went wrong, this should not be possible"
                )
        else:
            raise RuntimeError("Pruning went wrong, this should not be possible")
