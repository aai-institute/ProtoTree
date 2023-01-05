from abc import ABC, abstractmethod
from typing import Callable, TypeVar

import numpy as np
import torch
from pptree import print_tree
from torch import nn as nn
from torch.nn import functional as F

# TODO: a lot of stuff here is very poorly optimized, multiple time exponential complexity calls, even in properties

T = TypeVar("T")


# TODO: replace properties by methods, they are actually rather expensive to compute!
class Node(nn.Module, ABC):
    def __init__(self, index: int, parent: "InternalNode" = None):
        super().__init__()
        self.parent = parent
        self.index = index

    def get_path_from_start_node(self, start_node: "Node" = None):
        """
        :param start_node: if None, will start the path from the root node
        :return:
        """
        path = []
        max_path_len = self.depth() + 1

        def is_target(node):
            if start_node is None:
                return node.is_root
            return node == start_node

        found_target = False
        cur_node = self
        for i in range(max_path_len):
            path.append(cur_node)
            if is_target(cur_node):
                found_target = True
                break
            cur_node = cur_node.parent
        if not found_target:
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
    def sibling(self: T) -> T:
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

    def forward(self, xs: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}({self.index})"

    def print_tree(self, horizontal=True):
        print_tree(
            self, childattr="child_nodes", nameattr="name", horizontal=horizontal
        )

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @property
    def descendants(self) -> set["Node"]:
        return self.descendant_internal_nodes.union(self.leaves)

    @property
    @abstractmethod
    def leaves(self) -> set["Leaf"]:
        pass

    @property
    @abstractmethod
    def descendant_internal_nodes(self) -> set["InternalNode"]:
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

    def _lens_paths_to_leaves(self) -> list[int]:
        return [
            len(leaf.get_path_from_start_node(start_node=self)) for leaf in self.leaves
        ]

    def max_height(self) -> int:
        return max(self._lens_paths_to_leaves()) - 1

    def mean_hight(self):
        return np.mean(self._lens_paths_to_leaves()) - 1

    def min_height(self):
        return min(self._lens_paths_to_leaves()) - 1

    def depth(self):
        if self.is_root:
            return 0
        return self.parent.depth() + 1

    @property
    @abstractmethod
    def child_nodes(self) -> list["Node"]:
        pass


class InternalNode(Node):
    def __init__(
        self,
        index: int,
        left: Node = None,
        right: Node = None,
        parent: "InternalNode" = None,
        log_probabilities=False,
    ):
        super().__init__(index, parent=parent)
        self.left = left
        self.right = right

        # Flag that indicates whether probabilities or log probabilities are computed
        self.log_probabilities = log_probabilities

    # TODO: remove kwargs everywhere
    def forward(
        self,
        x: torch.Tensor,
        similarities: list[torch.Tensor] = None,
        out_map: dict[Node, int] = None,
        node_attr: dict[tuple[Node, str], torch.Tensor] = None,
    ):
        """

        :param x:
        :param similarities:
        :param out_map:
        :param node_attr: meant to assign attributes to nodes, like p_arrival and p_right. Passed down to
            children and modified by them...
        :return: stacked prob distributions of children, modified node_attr
        """
        batch_size = x.size(0)

        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        node_attr = {} if node_attr is None else node_attr
        # TODO: there was an if-else here but it did the same in both cases...
        # TODO: jesus, a dict with tuples of specific structure as keys!!! Rewrite
        p_arrival = node_attr.setdefault(
            (self, "p_arrival"), torch.ones(batch_size, device=x.device)
        )

        # Obtain the probabilities of taking the right subtree
        p_right = similarities[out_map[self]].squeeze(1)  # shape: (bs,)

        # TODO: this whole log-exp stuff has to be removed
        if self.log_probabilities:
            node_attr[self, "p_right"] = p_right

            # Store path probabilities of arriving at child nodes as node attributes
            # source: rewritten to pytorch from
            # https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/math/generic.py#L447-L471
            # add small epsilon for numerical stability
            _p_right = torch.abs(p_right) + 1e-7
            # TODO: what's up with log(2) here? Does this have to be so complicated?
            p_left = torch.where(
                _p_right < np.log(2),
                torch.log(-torch.expm1(-_p_right)),
                torch.log1p(-torch.exp(-_p_right)),
            )

            node_attr[self.left, "p_arrival"] = p_left + p_arrival
            node_attr[self.right, "p_arrival"] = p_right + p_arrival

            # TODO: node_attr is getting modified all the time, this is really bad practice and probably unnecessary
            # Obtain the probability distributions from the child nodes
            # shape: (bs, k)
            # TODO: maybe rewrite everything as for-loop over children,
            #  although generalization semantics is not entirely clear
            l_dists, _ = self.left.forward(
                x,
                similarities=similarities,
                out_map=out_map,
                node_attr=node_attr,
            )
            r_dists, _ = self.right.forward(
                x,
                similarities=similarities,
                out_map=out_map,
                node_attr=node_attr,
            )

            # Weight the probability distributions by this node's output
            p_right = p_right.view(batch_size, 1)
            p_left = p_left.view(batch_size, 1)
            pred_left_right_logits = torch.stack([p_left + l_dists, p_right + r_dists])
            # this is the log of a weighted sum of the distributions of the children
            # need logsumexp because everything is permanently switching between log and prop spaces...
            # TODO: simplify this once we only deal with logits or probs
            # shape: (bs, k)
            pred_logits = torch.logsumexp(pred_left_right_logits, dim=0)
            return pred_logits, node_attr
        else:
            raise NotImplementedError()
            # # Store decision node probabilities as node attribute
            # node_attr[self, "p_right"] = p_right
            # # Store path probabilities of arriving at child nodes as node attributes
            # node_attr[self.left, "p_arrival"] = (1 - p_right) * p_arrival
            # node_attr[self.right, "p_arrival"] = p_right * p_arrival
            # # # Store alpha value for this batch for this decision node
            # # node_attr[self, 'alpha'] = torch.sum(p_arrival * ps) / torch.sum(p_arrival)
            #
            # # Obtain the unweighted probability distributions from the child nodes
            # l_dists, _ = self.left.forward(
            #     xs,
            #     similarities=similarities,
            #     out_map=out_map,
            #     node_attr=node_attr,
            # )  # shape: (bs, k)
            # r_dists, _ = self.right.forward(
            #     xs,
            #     similarities=similarities,
            #     out_map=out_map,
            #     node_attr=node_attr,
            # )  # shape: (bs, k)
            # # Weight the probability distributions by the decision node's output
            # p_right = p_right.view(batch_size, 1)
            # # shape: (bs, k)
            # return (1 - p_right) * l_dists + p_right * r_dists, node_attr

    @property
    def size(self) -> int:
        return 1 + self.left.size + self.right.size

    # return all leaves of direct children
    @property
    def leaves(self) -> set["Leaf"]:
        return {leaf for child in self.child_nodes for leaf in child.leaves}

    @property
    def child_nodes(self) -> list["Node"]:
        return [self.left, self.right]

    @property
    def descendant_internal_nodes(self) -> set["InternalNode"]:
        return {
            self,
            *self.left.descendant_internal_nodes,
            *self.right.descendant_internal_nodes,
        }

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
        kontschieder_normalization=False,
        log_probabilities=False,
        disable_derivative_free_leaf_optim=False,
    ):
        super().__init__(index, parent=parent)

        if disable_derivative_free_leaf_optim:
            weights, requires_grad = torch.randn(num_classes), True
        elif kontschieder_normalization:
            weights, requires_grad = torch.ones(num_classes), False
        else:
            weights, requires_grad = torch.zeros(num_classes), False

        self.dist_params = nn.Parameter(weights, requires_grad=requires_grad)
        self.log_probabilities = log_probabilities
        self.kontschieder_normalization = kontschieder_normalization

    def predicted_label(self):
        return torch.argmax(self.distribution()).item()

    # TODO: remove kwargs? They are passed from parents, currently internal nodes and leaves have different signatures
    # TODO: IMPORTANT this doesn't compute anything, it just returns the stored distribution copied batch_size times.
    #  and performs a side-effect on the passed node_attr dict. This shouldn't be really called a forward, I think.
    #  On top of that, gradient based optimization of the leaf distribution doesn't seem to work anyway, so this
    #  thing doesn't even need to be a nn.Module. The whole node-index-dict-prototype-per-index-retrieval should
    #  probably be entirely rewritten.
    def forward(
        self, x: torch.Tensor, node_attr: dict = None, **kwargs
    ) -> tuple[torch.Tensor, dict]:

        batch_size = x.size(0)

        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        node_attr.setdefault(
            (self, "p_arrival"), torch.ones(batch_size, device=x.device)
        )

        dist = self.distribution().view(1, -1)  # shape: (1, k)
        # Same distribution for each x in the batch
        dists = torch.cat([dist] * batch_size, dim=0)  # shape: (bs, k)
        node_attr[self, "distribution"] = dists

        return dists, node_attr

    # TODO: simplify
    def distribution(self) -> torch.Tensor:
        if not self.kontschieder_normalization:
            if self.log_probabilities:
                result = F.log_softmax(self.dist_params, dim=0)
            else:
                # Return numerically stable softmax (see http://www.deeplearningbook.org/contents/numerical.html)
                result = F.softmax(
                    self.dist_params - torch.max(self.dist_params), dim=0
                )

        else:
            # kontschieder_normalization's version that uses a normalization factor instead of softmax:
            # TODO: overall kontschieder normalization is related to a paper implemented in
            #  https://github.com/mapillary/inplace_abn. Can it be just imported from there?
            #  Or maybe dropped altogether?
            if self.log_probabilities:
                result = torch.log(
                    (self.dist_params / torch.sum(self.dist_params)) + 1e-10
                )  # add small epsilon for numerical stability
            else:
                result = self.dist_params / torch.sum(self.dist_params)
        return result

    def y_proba(self):
        y_proba = self.distribution()
        if self.log_probabilities:
            y_proba = torch.exp(y_proba)
        return y_proba

    @property
    def requires_grad(self) -> bool:
        return self.dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self.dist_params.requires_grad = val

    @property
    def size(self) -> int:
        return 1

    @property
    def leaves(self) -> set:
        return {self}

    @property
    def child_nodes(self) -> list["Node"]:
        return []

    @property
    def descendant_internal_nodes(self) -> set:
        return set()

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
    log_probabilities=True,
    kontschieder_normalization=False,
    disable_derivative_free_leaf_optim=False,
):
    """
    Create a full binary tree with the given height.
    The leaves will carry distributions with the given number of classes.

    :param height:
    :param num_classes:
    :param log_probabilities:
    :param kontschieder_normalization:
    :param disable_derivative_free_leaf_optim:
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
            log_probabilities=log_probabilities,
            kontschieder_normalization=kontschieder_normalization,
            disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
        )

    root = InternalNode(0, log_probabilities=log_probabilities)

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

            node.left = InternalNode(
                left_index, parent=node, log_probabilities=log_probabilities
            )
            node.right = InternalNode(
                right_index, parent=node, log_probabilities=log_probabilities
            )

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
    max_height = root.max_height()
    cur_max_height = max_height
    cur_nodes = [root]
    for _ in range(max_height):
        next_nodes = []
        for node in cur_nodes:
            assert (
                node.depth() == cur_depth
            ), f"Node {node} has depth {node.depth} but should be {cur_depth}"
            node_max_height = node.max_height()
            node_min_height = node.min_height()
            assert (
                node_max_height <= cur_max_height
            ), f"Node {node} has height {node.height} but should be at most {cur_max_height}"
            if node.is_leaf:
                assert (
                    node_max_height == 0
                ), f"Leaf {node} has height {node.height} but should be 0"
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
    path_from_root = node.get_path_from_start_node()
    assert (
        path_from_root[0] == root
    ), f"Path from root should start with root but got: {path_from_root[0]}"
    assert (
        path_from_root[-1] == node
    ), f"Path from root should end with node but got: {path_from_root[-1]}"
    min_height = 0 if node.is_leaf else 1
    node_max_height = node.max_height()
    root_max_height = root.max_height()
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
    assert leaves.issubset(
        root.leaves
    ), f"Leaves of node {node} should be a subset of root leaves."
    for leaf in leaves:
        assert isinstance(leaf, Leaf), f"Leaf {leaf} is not of type Leaf: {type(leaf)}"
        assert (
            node in leaf.get_path_from_start_node()
        ), f"Node {node} should be in path from root to its leaf {leaf}"


def _health_check_root(root: InternalNode):
    assert (
        root.parent is None
    ), f"Root node should have no parent but got: {root.parent}"
    assert root.is_root, f"Root node {root} should be root but got: {root.is_root}"
    roots = [node for node in root.descendants if node.is_root]
    assert len(roots) == 1, f"Tree should have exactly one root but got: {roots}"


def health_check(root: InternalNode, height: int = None):
    """
    Checks that the tree is valid, i.e. that all nodes have the correct number of children and that the tree is
    connected. Raises an AssertionError if the tree is invalid.

    :param root: The root node of the tree to check.
    :param height: If passed, will also check that the tree has the correct height.
    """
    if height is not None:
        assert (
            root.max_height == height
        ), f"Root height should be {height} but got: {root.max_height}"

    _health_check_root(root)
    _health_check_height_depth(root)
    for node in root.descendants:
        _health_check_parent(node)
        _health_check_children(node)
        _health_check_connectivity(node, root)
