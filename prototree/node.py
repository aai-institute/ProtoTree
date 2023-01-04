from abc import ABC, abstractmethod
from copy import copy
from typing import TypeVar, Union

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

T = TypeVar("T", bound="NodeBase")


# TODO: replace properties by methods, they are actually rather expensive to compute!
class Node(nn.Module, ABC):
    def __init__(self, index: int):
        super().__init__()
        self.index = index

    def forward(self, xs: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        pass

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
        pass

    @property
    @abstractmethod
    def node_by_index(self) -> dict[int, "Node"]:
        pass

    @property
    def num_internal_nodes(self) -> int:
        return len(self.descendant_internal_nodes)

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    @property
    @abstractmethod
    def depth(self) -> int:
        pass

    @property
    @abstractmethod
    def child_nodes(self) -> list["Node"]:
        pass


class InternalNode(Node):
    def __init__(self, index: int, left: Node, right: Node, parent: "InternalNode" = None, log_probabilities=False):
        super().__init__(index)
        self.left = left
        self.right = right
        self.parent = parent

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
        return (
            {self}
            .union(self.left.descendant_internal_nodes)
            .union(self.right.descendant_internal_nodes)
        )

    @property
    def node_by_index(self) -> dict:
        return {
            self.index: self,
            **self.left.node_by_index,
            **self.right.node_by_index,
        }

    @property
    def num_internal_nodes(self) -> int:
        return 1 + self.left.num_internal_nodes + self.right.num_internal_nodes

    @property
    def num_leaves(self) -> int:
        return self.left.num_leaves + self.right.num_leaves

    @property
    def depth(self) -> int:
        return max(self.left.depth, self.right.depth) + 1


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
        Node.__init__(self, index)
        self.parent = parent

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
    def node_by_index(self) -> dict:
        return {self.index: self}

    @property
    def num_internal_nodes(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 0


def create_tree(
    depth: int,
    num_classes: int,
    log_probabilities=True,
    kontschieder_normalization=False,
    disable_derivative_free_leaf_optim=False,
):
    def create_tree_from_node_index(
        index: int, cur_depth: int
    ) -> Union[InternalNode, Leaf]:
        if cur_depth == depth:
            return Leaf(
                index,
                num_classes,
                kontschieder_normalization=kontschieder_normalization,
                log_probabilities=log_probabilities,
                disable_derivative_free_leaf_optim=disable_derivative_free_leaf_optim,
            )
        else:
            next_index = index + 1
            next_depth = cur_depth + 1
            left = create_tree_from_node_index(next_index, next_depth)
            right = create_tree_from_node_index(next_index + left.size, next_depth)
            return InternalNode(
                index,
                left,
                right,
                log_probabilities=log_probabilities,
            )

    return create_tree_from_node_index(0, 0)


class Tree(nn.Module):
    def __init__(self, depth, num_classes, **kwargs):
        super().__init__()
        self.root = create_tree(depth, num_classes, **kwargs)