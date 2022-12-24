from abc import ABC, abstractmethod
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

    # TODO: don't descendants already contain leaves? Seems like this method is redundant.
    @property
    def nodes(self) -> set["Node"]:
        return self.descendants.union(self.leaves)

    @property
    @abstractmethod
    def leaves(self) -> set["Node"]:
        pass

    @property
    @abstractmethod
    def descendants(self) -> set["Node"]:
        pass

    @property
    @abstractmethod
    def descendants_by_index(self) -> dict[int, "Node"]:
        pass

    @property
    def num_descendants(self) -> int:
        return len(self.descendants)

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    @property
    @abstractmethod
    def depth(self) -> int:
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

    def forward(self, xs: torch.Tensor, **kwargs):

        # Get the batch size
        batch_size = xs.size(0)

        # Keep a dict to assign attributes to nodes. Create one if not already existent
        # TODO: why is this necessary? Maybe remove, forward signature is messed up
        node_attr = kwargs.setdefault("attr", dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        if not self.log_probabilities:
            p_arrival = node_attr.setdefault(
                (self, "p_arrival"), torch.ones(batch_size, device=xs.device)
            )
        else:
            p_arrival = node_attr.setdefault(
                (self, "p_arrival"), torch.ones(batch_size, device=xs.device)
            )

        # Obtain the probabilities of taking the right subtree
        p_stop = self.g(xs, **kwargs)  # shape: (bs,)

        if not self.log_probabilities:
            # Store decision node probabilities as node attribute
            node_attr[self, "p_stop"] = p_stop
            # Store path probabilities of arriving at child nodes as node attributes
            node_attr[self.left, "p_arrival"] = (1 - p_stop) * p_arrival
            node_attr[self.right, "p_arrival"] = p_stop * p_arrival
            # # Store alpha value for this batch for this decision node
            # node_attr[self, 'alpha'] = torch.sum(p_arrival * ps) / torch.sum(p_arrival)

            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.left.forward(xs, **kwargs)  # shape: (bs, k)
            r_dists, _ = self.right.forward(xs, **kwargs)  # shape: (bs, k)
            # Weight the probability distributions by the decision node's output
            p_stop = p_stop.view(batch_size, 1)
            return (
                1 - p_stop
            ) * l_dists + p_stop * r_dists, node_attr  # shape: (bs, k)
        else:
            # Store decision node probabilities as node attribute
            node_attr[self, "p_stop"] = p_stop

            # Store path probabilities of arriving at child nodes as node attributes
            # source: rewritten to pytorch from
            # https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/math/generic.py#L447-L471
            x = torch.abs(p_stop) + 1e-7  # add small epsilon for numerical stability
            oneminusp = torch.where(
                x < np.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x))
            )

            node_attr[self.left, "p_arrival"] = oneminusp + p_arrival
            node_attr[self.right, "p_arrival"] = p_stop + p_arrival

            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.left.forward(xs, **kwargs)  # shape: (bs, k)
            r_dists, _ = self.right.forward(xs, **kwargs)  # shape: (bs, k)

            # Weight the probability distributions by the decision node's output
            p_stop = p_stop.view(batch_size, 1)
            oneminusp = oneminusp.view(batch_size, 1)
            logs_stacked = torch.stack((oneminusp + l_dists, p_stop + r_dists))
            return torch.logsumexp(logs_stacked, dim=0), node_attr  # shape: (bs,)

    def g(self, xs: torch.Tensor, **kwargs):
        out_map = kwargs[
            "out_map"
        ]  # Obtain the mapping from decision nodes to conv net outputs
        conv_net_output = kwargs["conv_net_output"]  # Obtain the conv net outputs
        out = conv_net_output[
            out_map[self]
        ]  # Obtain the output corresponding to this decision node
        return out.squeeze(dim=1)

    @property
    def size(self) -> int:
        return 1 + self.left.size + self.right.size

    @property
    def leaves(self) -> set:
        return self.left.leaves.union(self.right.leaves)

    @property
    def descendants(self) -> set:
        return {self}.union(self.left.descendants).union(self.right.descendants)

    @property
    def descendants_by_index(self) -> dict:
        return {
            self.index: self,
            **self.left.descendants_by_index,
            **self.right.descendants_by_index,
        }

    @property
    def num_descendants(self) -> int:
        return 1 + self.left.num_descendants + self.right.num_descendants

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
            self._dist_params = nn.Parameter(
                torch.randn(num_classes), requires_grad=True
            )
        elif kontschieder_normalization:
            self._dist_params = nn.Parameter(
                torch.ones(num_classes), requires_grad=False
            )
        else:
            self._dist_params = nn.Parameter(
                torch.zeros(num_classes), requires_grad=False
            )

        # Flag that indicates whether probabilities or log probabilities are computed
        self.log_probabilities = log_probabilities
        self.kontschieder_normalization = kontschieder_normalization

    @property
    def dist_params(self) -> torch.Tensor:
        return self._dist_params

    def forward(self, xs: torch.Tensor, **kwargs):

        # Get the batch size
        batch_size = xs.size(0)

        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault("attr", dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        if not self.log_probabilities:
            node_attr.setdefault(
                (self, "p_arrival"), torch.ones(batch_size, device=xs.device)
            )
        else:
            node_attr.setdefault(
                (self, "p_arrival"), torch.zeros(batch_size, device=xs.device)
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
                return F.log_softmax(self._dist_params, dim=0)
            else:
                # Return numerically stable softmax (see http://www.deeplearningbook.org/contents/numerical.html)
                return F.softmax(
                    self._dist_params - torch.max(self._dist_params), dim=0
                )

        else:
            # kontschieder_normalization's version that uses a normalization factor instead of softmax:
            if self.log_probabilities:
                return torch.log(
                    (self._dist_params / torch.sum(self._dist_params)) + 1e-10
                )  # add small epsilon for numerical stability
            else:
                return self._dist_params / torch.sum(self._dist_params)

    @property
    def requires_grad(self) -> bool:
        return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    @property
    def size(self) -> int:
        return 1

    @property
    def leaves(self) -> set:
        return {self}

    @property
    def descendants(self) -> set:
        return set()

    @property
    def descendants_by_index(self) -> dict:
        return {self.index: self}

    @property
    def num_descendants(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 0
