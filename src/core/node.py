import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, TypeVar, Union

import numpy as np
import torch
from pptree import print_tree
from torch import nn as nn
from torch.nn import functional as F

from src.util.math import log1mexp

# TODO: A lot of stuff here is very poorly optimized, multiple calls with exponential time complexity in the depth,
#  even in properties.

TNode = TypeVar("TNode", bound="Node")
log = logging.getLogger(__name__)


class Node(ABC):
    # TODO: Replace properties by methods, they are actually rather expensive to compute. Perhaps we can cache them?
    # TODO: There's large number of methods who depend on subclasses,
    #  https://luzkan.github.io/smells/base-class-depends-on-subclass

    def __init__(self, index: int, parent: Optional["InternalNode"] = None):
        super().__init__()
        self.parent = parent
        self.index = index

    def get_path_from_ancestor(
        self, start_node: Optional["Node"] = None
    ) -> list["Node"]:
        """
        Returns a path from the selected start_node to self.

        :param start_node: if None, the resulting path will start from the root node
        :return:
        """
        path = []
        max_path_len = self.depth + 1

        def is_start_node(node):
            if start_node is None:
                return node.is_root
            return node == start_node

        found_start_node = False
        cur_node = self
        for _ in range(max_path_len):
            path.append(cur_node)
            if is_start_node(cur_node):
                found_start_node = True
                break
            cur_node = cur_node.parent
        if not found_start_node:
            raise ValueError(
                f"No path found from {start_node} to {self}! Check that the start node is in the tree"
            )
        return path[::-1]

    @property
    def is_root(self) -> bool:
        return self.parent is None

    def get_root(self):
        cur_node = self
        while True:
            if cur_node.is_root:
                return cur_node
            cur_node = cur_node.parent

    @property
    def is_left_child(self) -> bool:
        if self.is_root:
            return False
        return self.parent.left == self

    @property
    def is_right_child(self) -> bool:
        if self.is_root:
            return False
        return self.parent.right == self

    @property
    def sibling(self: TNode) -> TNode:
        return self.parent.left if self.is_right_child else self.parent.right

    @property
    def is_leaf(self):
        return len(self.child_nodes) == 0

    @property
    def is_internal(self):
        return not self.is_leaf

    @property
    def name(self):
        return str(self.index)

    @abstractmethod
    def forward(
        self, node_to_probs: dict[Union["InternalNode", "Leaf"], "NodeProbabilities"]
    ) -> torch.Tensor:
        """
        Computes the predicted logits for the batch

        :param node_to_probs:
        :return: tensor of shape (batch_size, k)
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.index})"

    def print_tree(self, horizontal=True):
        print_tree(
            self, childattr="child_nodes", nameattr="name", horizontal=horizontal
        )

    @property
    def size(self) -> int:
        return len(self.descendants)

    @property
    def descendants(self) -> list[Union["Leaf", "InternalNode"]]:
        return self.descendant_internal_nodes + self.leaves

    @property
    def ancestors(self) -> list["InternalNode"]:
        """
        :return: The ancestors of the node, starting from the root.
        """
        cur_node = self
        ancestors = []
        while not cur_node.is_root:
            cur_node = cur_node.parent
            ancestors.append(cur_node)
        return ancestors[::-1]

    @property
    @abstractmethod
    def leaves(self) -> list["Leaf"]:
        pass

    @property
    @abstractmethod
    def descendant_internal_nodes(self) -> list["InternalNode"]:
        """
        Including self, if appropriate.
        """
        pass

    def get_idx2node(self) -> dict[int, "Node"]:
        return {node.index: node for node in self.descendants}

    @property
    def num_internal_nodes(self) -> int:
        return len(self.descendant_internal_nodes)

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    @property
    def _lens_paths_to_leaves(self) -> list[int]:
        return [
            len(leaf.get_path_from_ancestor(start_node=self)) for leaf in self.leaves
        ]

    @property
    def max_height(self) -> int:
        return max(self._lens_paths_to_leaves) - 1

    @property
    def mean_height(self):
        return np.mean(self._lens_paths_to_leaves) - 1

    @property
    def min_height(self):
        return min(self._lens_paths_to_leaves) - 1

    @property
    def depth(self):
        if self.is_root:
            return 0
        return self.parent.depth + 1

    @property
    @abstractmethod
    def child_nodes(self) -> list["Node"]:
        pass


class InternalNode(Node):
    def __init__(
        self,
        index: int,
        left: Union["InternalNode", "Leaf"] = None,
        right: Union["InternalNode", "Leaf"] = None,
        parent: Optional["InternalNode"] = None,
    ):
        super().__init__(index, parent=parent)
        # not optional b/c in a healthy tree, every node has a left and right child
        # they can be passed as None in init b/c it is more convenient to create a tree this way,
        # see create_tree implementation
        self.left = left
        self.right = right

    # TODO: remove kwargs everywhere, make leaf and internal node forwards consistent
    def forward(
        self,
        node_to_probs: dict[Union["InternalNode", "Leaf"], "NodeProbabilities"],
    ):
        probs = node_to_probs[self]

        # shape: (bs, k)
        l_logits = self.left.forward(node_to_probs)
        r_logits = self.right.forward(node_to_probs)

        # Weight the probability distributions by this node's output
        log_p_right = probs.log_p_right.unsqueeze(-1)
        log_p_left = probs.log_p_left.unsqueeze(-1)

        pred_left_right_logits = torch.stack(
            [log_p_left + l_logits, log_p_right + r_logits]
        )
        # this is the log of a weighted sum of the predicted prob. distributions of the children
        # i.e. log(p_left * exp(l_logits) + p_right * exp(r_logits))
        # shape: (bs, k)
        mean_logits = torch.logsumexp(pred_left_right_logits, dim=0)
        return mean_logits

    # return all leaves of direct children
    @property
    def leaves(self) -> list["Leaf"]:
        return [leaf for child in self.child_nodes for leaf in child.leaves]

    @property
    def child_nodes(self) -> list[Union["InternalNode", "Leaf"]]:
        return [self.left, self.right]

    @property
    def descendant_internal_nodes(self) -> list["InternalNode"]:
        result = [self]
        for child in self.child_nodes:
            result.extend(child.descendant_internal_nodes)
        return result

    @property
    def num_internal_nodes(self) -> int:
        return 1 + self.left.num_internal_nodes + self.right.num_internal_nodes

    @property
    def num_leaves(self) -> int:
        return self.left.num_leaves + self.right.num_leaves


class Leaf(Node):
    def __init__(
        self,
        index: int,
        num_classes: int,
        parent: InternalNode = None,
        gradient_opt: bool = False,
    ):
        super().__init__(index, parent=parent)

        self.num_classes = num_classes
        self.dist_params: nn.Parameter = nn.Parameter(
            torch.randn(num_classes) * 1e-3, requires_grad=gradient_opt
        )
        if not gradient_opt:
            self.dist_param_update_count = 0

    def to(self, *args, **kwargs):
        self.dist_params = self.dist_params.to(*args, **kwargs)
        return self

    def predicted_label(self) -> int:
        return self.y_logits().argmax().item()

    def conf_predicted_label(self) -> float:
        return self.y_probs().max().item()

    # Note: this doesn't compute anything, it just returns the stored distribution copied batch_size times.
    def forward(
        self, node_to_probs: dict[Union["InternalNode", "Leaf"], "NodeProbabilities"]
    ) -> torch.Tensor:
        return self.y_logits_batch(node_to_probs[self].batch_size)

    def y_logits_batch(self, batch_size: int) -> torch.Tensor:
        logits = self.y_logits()
        return logits.unsqueeze(0).repeat(batch_size, 1)

    def y_logits(self) -> torch.Tensor:
        return F.log_softmax(self.dist_params, dim=0)

    def y_probs(self):
        return torch.exp(self.y_logits())

    @property
    def leaves(self) -> list["Leaf"]:
        return [self]

    @property
    def child_nodes(self) -> list[Union["InternalNode", "Leaf"]]:
        return []

    @property
    def descendant_internal_nodes(self) -> list["InternalNode"]:
        return []

    @property
    def num_internal_nodes(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1


# TODO: kinda ugly, setting left and right and the leaves explicitly. However, an improvement would require
#  significant refactoring of the code. It is not clear whether a generalization would be helpful,
#  as several concepts hinge on the binary tree structure.
def create_tree(
    height: int,
    num_classes: int,
    gradient_leaf_opt: bool = False,
):
    """
    Create a full binary tree with the given height.
    The leaves will carry distributions with the given number of classes.

    :return: the root node of the created tree
    """
    if height < 1:
        raise ValueError(f"Depth must be at least 1 but got: {height}")
    if num_classes < 2:
        raise ValueError(f"Number of classes must be at least 2 but got: {num_classes}")

    def get_leaf(index, parent):
        return Leaf(
            index,
            num_classes,
            parent=parent,
            gradient_opt=gradient_leaf_opt,
        )

    root = InternalNode(0)

    # create tree of internal nodes
    cur_nodes = [root]
    for cur_height in range(1, height):
        cur_nodes_size = 2 ** (height - cur_height + 1) - 1
        next_nodes = []
        for node in cur_nodes:
            # TODO: see todo above about size and index. Here size is the size of the subtree rooted at the node
            #  I am not sure this is needed at all but keeping it for now as index is used in visualization,
            #  index is based on size and I am afraid of touching it now. Note that the size property is actually
            #  not static, so after pruning the sizes do not match the original tree.
            #  But here we create a full tree, so it's ok. Also, the size is actually not used anywhere.
            left_index = node.index + 1
            # TODO: reproducing index logic from before, see todo above
            right_index = left_index + cur_nodes_size

            node.left = InternalNode(left_index, parent=node)
            node.right = InternalNode(right_index, parent=node)

            next_nodes.extend([node.left, node.right])
        cur_nodes = next_nodes

    # Add leaves to last layer
    for node in cur_nodes:
        left_index = node.index + 1
        right_index = left_index + 1
        node.left = get_leaf(left_index, node)
        node.right = get_leaf(right_index, node)
    return root


def reindex_tree(root: InternalNode):
    """
    Reindex the tree in a height-first manner, will create the same indexing as the `create_tree` function.

    I introduced this method to have a temporary fix for
    dealing with indices. In is not really used atm but it at least serves to indicate the indexing
    logic and being able to split off trees with the same index semantics.
    TODO: see other TODOs about indices.
    """

    def _reindex(node, index=0):
        node.index = index
        for i, child in enumerate(node.child_nodes):
            _reindex(child, index=index + 1 + i * child.size)

    _reindex(root)


def remove_from_tree(node: Node):
    """
    Removes a node and its parent from the binary tree in which they reside
    by connecting the grandparent to its sibling. Cannot be applied to the root node or its direct children.
    Important: this modifies the tree inplace!
    """
    if node.parent is None:
        raise ValueError(f"Cannot remove root node (no parent). Node: {node}")
    grandparent = node.parent.parent
    if grandparent is None:
        raise ValueError(
            f"Cannot remove this node as its parent is the root and it therefore has no grandparent. Node: {node}"
        )
    sibling = node.sibling

    # creating a direct connection between grandparent and sibling, and removing node, subtree and parent
    sibling.parent = grandparent
    if node.parent.is_left_child:
        grandparent.left = sibling
    else:
        grandparent.right = sibling

    # TODO: the removed node becomes the root of a subtree when parent is None,
    #  but indices are not updated. Fix when indices are dealt with or removed altogether.
    node.parent = None


def get_max_height_nodes(root: Node, condition: Callable[[Node], bool]) -> list[Node]:
    """
    Returns a list of deepest-possible nodes that satisfy the given condition, assuming that the condition
    automatically holds on all child nodes if it holds on the parent node. For example, most leaves-based conditions
    are of this type.
    """
    result = []

    def append_children(node: Node):
        if all(condition(leaf) for leaf in node.leaves):
            result.append(node)
        else:
            for child in node.child_nodes:
                append_children(child)

    append_children(root)
    return result


def _health_check_height_depth(root: Node):
    """
    Checks that the heights and depths of the nodes in the tree are consistent.
    The max_height of a node is the number of edges on the longest path from that node to a leaf.
    The depth of a node is the number of edges from the root to that node.
    """
    cur_depth = 0
    max_height = root.max_height
    cur_max_height = max_height
    cur_nodes = [root]
    for _ in range(max_height):
        next_nodes = []
        for node in cur_nodes:
            assert (
                node.depth == cur_depth
            ), f"Node {node} has depth {node.depth} but should be {cur_depth}"
            node_max_height = node.max_height
            node_min_height = node.min_height
            assert (
                node_max_height <= cur_max_height
            ), f"Node {node} has max_height {node.max_height} but should be at most {cur_max_height}"
            if node.is_leaf:
                assert (
                    node_max_height == 0
                ), f"Leaf {node} has max_height {node.max_height} but should be 0"
            else:
                assert (
                    node_min_height >= 1
                ), f"Node {node} has min height {node_min_height} but should be at least 1"
            next_nodes.extend(node.child_nodes)
        cur_nodes = next_nodes
        cur_depth += 1
        cur_max_height -= 1


def _health_check_parent(node: Node):
    if node.is_root:
        assert (
            node.parent is None
        ), f"Root node should have no parent but got: {node.parent}. Node: {node}"
    elif node.is_left_child:
        assert (
            node.parent.left == node
        ), f"Left child should have itself as parent's left child but got: {node.parent.left}. Node: {node}"
        assert (
            node.parent.right == node.sibling
        ), f"Left child should have parent's right child as sibling but got: {node.parent.right}. Node: {node}"
    else:
        assert (
            node.is_right_child
        ), f"Node should be either root, left or right child but got neither. Node: {node}."
        assert (
            node.parent.right == node
        ), f"Right child should have itself as parent's right child but got: {node.parent.right}. Node: {node}"
        assert (
            node.parent.left == node.sibling
        ), f"Right child should have parent's left child as sibling but got: {node.parent.left}. Node: {node}"


def _health_check_children(node: Node):
    if node.is_leaf:
        assert isinstance(
            node, Leaf
        ), f"Leaf node {node} is not of type Leaf: {type(node)}"
        assert (
            node.child_nodes == []
        ), f"Leaf node {node} has children: {node.child_nodes}"
    else:
        assert (
            node.is_internal
        ), f"Not leaf node should be internal but got: {node.is_internal}. Node: {node}"
        assert isinstance(
            node, InternalNode
        ), f"Node with children {node} is not of type InternalNode: {type(node)}"
        assert (
            len(node.child_nodes) == 2
        ), f"Node {node} has {len(node.child_nodes)} children, expected 2"
        assert (
            node.left.parent == node.right.parent == node
        ), f"Node {node} has children with wrong parent"


def _health_check_connectivity(node: Node, root: InternalNode):
    path_from_root = node.get_path_from_ancestor()
    assert (
        path_from_root[0] == root
    ), f"Path from root should start with root but got: {path_from_root[0]}"
    assert (
        path_from_root[-1] == node
    ), f"Path from root should end with node but got: {path_from_root[-1]}"
    min_height = 0 if node.is_leaf else 1
    node_max_height = node.max_height
    root_max_height = root.max_height
    assert (
        min_height <= node_max_height <= root_max_height
    ), f"Node max_height should be between 1 and root max_height ({root_max_height}) but got: {node_max_height}"

    # TODO: this is not height but rather height. Rename and explain.
    # could be shorter if tree is not full
    max_path_len = root_max_height - node_max_height + 1
    assert (
        len(path_from_root) <= max_path_len
    ), f"Path from root should be of length {max_path_len} but got: {len(path_from_root)}. Node: {node}"
    leaves = node.leaves
    assert set(leaves).issubset(
        root.leaves
    ), f"Leaves of node {node} should be a subset of root leaves."
    for leaf in leaves:
        assert isinstance(leaf, Leaf), f"Leaf {leaf} is not of type Leaf: {type(leaf)}"
        assert (
            node in leaf.get_path_from_ancestor()
        ), f"Node {node} should be in path from root to its leaf {leaf}"


def _health_check_root(root: InternalNode):
    assert (
        root.parent is None
    ), f"Root node should have no parent but got: {root.parent}"
    assert root.is_root, f"Root node {root} should be root but got: {root.is_root}"
    roots = [node for node in root.descendants if node.is_root]
    assert len(roots) == 1, f"Tree should have exactly one root but got: {roots}"


def health_check(root: InternalNode, max_height: int = None):
    """
    Checks that the tree is valid, i.e. that all nodes have the correct number of children and that the tree is
    connected. Raises an AssertionError if the tree is invalid.

    :param root: The root node of the tree to check.
    :param max_height: If passed, will also check that the tree has the correct height.
    """
    if max_height is not None:
        assert (
            root.max_height <= max_height
        ), f"Root max_height should be {max_height} but got: {root.max_height}"

    _health_check_root(root)
    _health_check_height_depth(root)
    for node in root.descendants:
        _health_check_parent(node)
        _health_check_children(node)
        _health_check_connectivity(node, root)


@dataclass
class NodeProbabilities:
    """
    Will hold the probabilities for each node for a batch of inputs. The probabilities are computed
    from prototypes similarities and the tree structure.
    All fields are tensors of shape (batch_size, ) or None (e.g. log_p_left/right for leaves).
    """

    log_p_arrival: torch.Tensor
    log_p_right: Optional[torch.Tensor] = None

    @property
    def log_p_left(self) -> Optional[torch.Tensor]:
        if self.log_p_right is None:
            return None
        return log1mexp(self.log_p_right)

    @property
    def log_p_arrival_left(self) -> Optional[torch.Tensor]:
        if self.log_p_right is None:
            return None
        return self.log_p_arrival + self.log_p_left

    @property
    def log_p_arrival_right(self) -> Optional[torch.Tensor]:
        if self.log_p_right is None:
            return None
        return self.log_p_arrival + self.log_p_right

    @property
    def batch_size(self) -> int:
        return self.log_p_arrival.shape[0]
