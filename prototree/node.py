from abc import ABC, abstractmethod
from copy import copy
from typing import TypeVar

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

T = TypeVar("T", bound="NodeBase")


# TODO: replace properties by methods, they are actually rather expensive to compute!
class Node(nn.Module, ABC):
    def __init__(self, index: int):
        super().__init__()
        self._index = index

    def forward(self, xs: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        pass

    @property
    def index(self) -> int:
        return self._index

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
    def __init__(
        self, index: int, left: Node, right: Node, log_probabilities: bool = False
    ):
        super().__init__(index)
        self.left = left
        self.right = right

        # Flag that indicates whether probabilities or log probabilities are computed
        self.log_probabilities = log_probabilities

    # TODO: remove kwargs everywhere
    def forward(
        self,
        xs: torch.Tensor,
        similarities: list[torch.Tensor] = None,
        out_map: dict[Node, int] = None,
        node_attr: dict = None,
    ):
        """

        :param xs:
        :param similarities:
        :param out_map:
        :param node_attr: meant to assign attributes to nodes, like p_arrival and p_right. Passed down to
            children and modified by them...
        :return:
        """

        # Get the batch size
        batch_size = xs.size(0)

        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        node_attr = {} if node_attr is None else node_attr
        # TODO: there was an if-else here but it did the same in both cases...
        p_arrival = node_attr.setdefault(
            (self, "p_arrival"), torch.ones(batch_size, device=xs.device)
        )

        # Obtain the probabilities of taking the right subtree
        p_right = similarities[out_map[self]].squeeze(1)  # shape: (bs,)

        if self.log_probabilities:
            node_attr[self, "p_right"] = p_right

            # Store path probabilities of arriving at child nodes as node attributes
            # source: rewritten to pytorch from
            # https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/math/generic.py#L447-L471
            x = torch.abs(p_right) + 1e-7  # add small epsilon for numerical stability
            oneminusp = torch.where(
                x < np.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x))
            )

            node_attr[self.left, "p_arrival"] = oneminusp + p_arrival
            node_attr[self.right, "p_arrival"] = p_right + p_arrival

            # TODO: node_attr is getting modified all the time, this is really bad practice and probably unnecessary
            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.left.forward(
                xs,
                similarities=similarities,
                out_map=out_map,
                node_attr=node_attr,
            )  # shape: (bs, k)
            r_dists, _ = self.right.forward(
                xs,
                similarities=similarities,
                out_map=out_map,
                node_attr=node_attr,
            )  # shape: (bs, k)

            # Weight the probability distributions by the decision node's output
            p_right = p_right.view(batch_size, 1)
            oneminusp = oneminusp.view(batch_size, 1)
            logs_stacked = torch.stack((oneminusp + l_dists, p_right + r_dists))
            return torch.logsumexp(logs_stacked, dim=0), node_attr  # shape: (bs,)
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
    def descendant_internal_nodes(self) -> set:
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
        kontschieder_normalization=False,
        log_probabilities=False,
        disable_derivative_free_leaf_optim=False,
    ):
        super().__init__(index)

        # Initialize the distribution parameters
        if disable_derivative_free_leaf_optim:
            self.dist_params = nn.Parameter(
                torch.randn(num_classes), requires_grad=True
            )
        elif kontschieder_normalization:
            self.dist_params = nn.Parameter(
                torch.ones(num_classes), requires_grad=False
            )
        else:
            self.dist_params = nn.Parameter(
                torch.zeros(num_classes), requires_grad=False
            )

        # Flag that indicates whether probabilities or log probabilities are computed
        self.log_probabilities = log_probabilities
        self.kontschieder_normalization = kontschieder_normalization

    def predicted_label(self):
        return torch.argmax(self.distribution()).item()

    # TODO: remove kwargs? They are passed from parents, currently internal nodes and leaves have different signatures
    def forward(
        self, xs: torch.Tensor, node_attr: dict = None, **kwargs
    ) -> torch.Tensor:

        # Get the batch size
        batch_size = xs.size(0)

        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        node_attr.setdefault(
            (self, "p_arrival"), torch.ones(batch_size, device=xs.device)
        )

        # Obtain the leaf distribution
        dist = self.distribution()  # shape: (k,)
        # Reshape the distribution to a matrix with one single row
        dist = dist.view(1, -1)  # shape: (1, k)
        # Duplicate the row for all x in xs
        dists = torch.cat((dist,) * batch_size, dim=0)  # shape: (bs, k)

        # Store leaf distributions as node property
        node_attr[self, "ds"] = dists

        # Return both the result of the forward pass as well as the node properties
        return dists, node_attr

    def distribution(self) -> torch.Tensor:
        if not self.kontschieder_normalization:
            if self.log_probabilities:
                return F.log_softmax(self.dist_params, dim=0)
            else:
                # Return numerically stable softmax (see http://www.deeplearningbook.org/contents/numerical.html)
                return F.softmax(self.dist_params - torch.max(self.dist_params), dim=0)

        else:
            # kontschieder_normalization's version that uses a normalization factor instead of softmax:
            if self.log_probabilities:
                return torch.log(
                    (self.dist_params / torch.sum(self.dist_params)) + 1e-10
                )  # add small epsilon for numerical stability
            else:
                return self.dist_params / torch.sum(self.dist_params)

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
