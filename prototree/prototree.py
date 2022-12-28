import pickle
from copy import copy
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn

from prototree.node import InternalNode, Leaf, Node
from util.func import min_pool2d
from util.l2conv import L2Conv2D


# TODO: inherit from Node?
class ProtoTree(nn.Module):
    ARGUMENTS = ["depth", "num_features", "W1", "H1", "log_probabilities"]

    SAMPLING_STRATEGIES = ["distributed", "sample_max", "greedy"]

    def __init__(
        self,
        num_classes: int,
        depth: int,
        out_channels: int,
        W1: int,
        H1: int,
        feature_net: torch.nn.Module,
        log_probabilities: bool = True,
        kontschieder_normalization: bool = False,
        kontschieder_train=False,
        add_on_layers: nn.Module = None,
        disable_derivative_free_leaf_optim=False,
    ):
        super().__init__()
        assert depth > 0
        assert num_classes > 0

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

        self._root = create_tree_from_node_index(0, 0)

        self._num_classes = num_classes

        self.out_channels = out_channels
        self.num_prototypes = self.num_internal_nodes

        # Keep a dict that stores a reference to each node's parent
        # Key: node -> Value: the node's parent
        # The root of the tree is mapped to None
        self.node2parent: dict[Node, Optional[InternalNode]] = {}
        self._set_parents()  # Traverse the tree to build the self._parents dict

        # Set the feature network
        self._net = feature_net
        self._add_on = add_on_layers if add_on_layers is not None else nn.Identity()

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = log_probabilities

        # Flag that indicates whether a normalization factor should be used instead of softmax.
        self._kontschieder_normalization = kontschieder_normalization
        self._kontschieder_train = kontschieder_train
        # Map each decision node to an output of the feature net
        # TODO: that's not really true, and why do we need this at all? Can't we use node id?
        #   This is used for retrieving outputs from some indexed list...
        self.out_map: dict[Node, int] = {
            n: i for i, n in zip(range(2**depth - 1), self.internal_nodes)
        }

        self.prototype_layer = L2Conv2D(self.num_prototypes, self.out_channels, W1, H1)

    def init_prototype_layer(self):
        # TODO: wassup with the constants?
        torch.nn.init.normal_(self.prototype_layer.prototype_vectors, mean=0.5, std=0.1)

    def device(self):
        return next(self.parameters()).device

    @property
    def prototype_shape(self):
        return self.prototype_layer.prototype_shape

    @property
    def root(self) -> InternalNode:
        return self._root

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def kontschieder_normalization(self):
        return self._kontschieder_normalization

    @property
    def log_probabilities(self) -> bool:
        return self._log_probabilities

    @property
    def add_on(self):
        return self._add_on

    @property
    def net(self):
        return self._net

    @property
    def root(self) -> Node:
        return self._root

    @property
    def leaves_require_grad(self) -> bool:
        return any([leaf.requires_grad for leaf in self.leaves])

    @leaves_require_grad.setter
    def leaves_require_grad(self, val: bool):
        for leaf in self.leaves:
            leaf.requires_grad = val

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def extract_features(self, x: torch.Tensor):
        x = self.net(x)
        x = self.add_on(x)
        return x

    def forward(
        self,
        xs: torch.Tensor,
        sampling_strategy: str = SAMPLING_STRATEGIES[0],  # `distributed` by default
    ) -> tuple:
        assert (
            sampling_strategy in ProtoTree.SAMPLING_STRATEGIES
        ), f"Got sampling_strategy={sampling_strategy}, expected one of {ProtoTree.SAMPLING_STRATEGIES}"

        # Perform a forward pass with the conv net
        features = self.extract_features(xs)
        bs, D, W, H = features.shape

        """
            COMPUTE THE PROTOTYPE SIMILARITIES GIVEN THE COMPUTED FEATURES
        """

        # Use the features to compute the distances from the prototypes
        # Shape: (batch_size, num_prototypes, W, H)
        distances = self.prototype_layer(features)

        # Perform global min pooling to see the minimal distance for each prototype to any patch of the input image
        min_distances = min_pool2d(distances, kernel_size=(W, H))
        min_distances = min_distances.view(bs, self.num_prototypes)

        similarities = -min_distances
        if not self.log_probabilities:
            similarities = torch.exp(similarities)

        # Split (or chunk) the conv net output tensor of shape (batch_size, num_decision_nodes) into individual tensors
        # of shape (batch_size, 1) containing the logits that are relevant to single decision nodes
        # TODO: here is something important, tightly coupled with out_map construction
        #   This turns it into a list. Do we need chunk? Can't we just use it directly?
        #   The variable should be renamed to something more meaningful.
        similarities = similarities.chunk(similarities.size(1), dim=1)
        # Add the mapping of decision nodes to conv net outputs to the kwargs dict to be passed to the decision nodes in
        # the tree. Use a copy, as the original should not be modified
        out_map = copy(self.out_map)

        """
            PERFORM A FORWARD PASS THROUGH THE TREE GIVEN THE COMPUTED SIMILARITIES
        """

        # Perform a forward pass through the tree
        out, attr = self.root.forward(xs, similarities=similarities, out_map=out_map)

        info = dict()
        # Store the probability of arriving at all nodes in the decision tree
        info["pa_tensor"] = {
            n.index: attr[n, "p_arrival"].unsqueeze(1) for n in self.all_nodes
        }
        # Store the output probabilities of all decision nodes in the tree
        info["p_right"] = {
            n.index: attr[n, "p_right"].unsqueeze(1) for n in self.internal_nodes
        }

        # Generate the output based on the chosen sampling strategy
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[0]:  # Distributed
            return out, info
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[1]:  # Sample max
            # Get the batch size
            batch_size = xs.size(0)
            # Get an ordering of all leaves in the tree
            leaves = list(self.leaves)
            # Obtain path probabilities of arriving at each leaf
            pas = [
                attr[l, "p_arrival"].view(batch_size, 1) for l in leaves
            ]  # All shaped (bs, 1)
            # Obtain output distributions of each leaf
            dss = [
                attr[l, "ds"].view(batch_size, 1, self._num_classes) for l in leaves
            ]  # All shaped (bs, 1, k)
            # Prepare data for selection of most probable distributions
            # Let L denote the number of leaves in this tree
            pas = torch.cat(tuple(pas), dim=1)  # shape: (bs, L)
            dss = torch.cat(tuple(dss), dim=1)  # shape: (bs, L, k)
            # Select indices (in the 'leaves' variable) of leaves with highest path probability
            ix = torch.argmax(pas, dim=1).long()  # shape: (bs,)
            # Select distributions of leafs with highest path probability
            dists = []
            for j, i in zip(range(dss.shape[0]), ix):
                dists += [dss[j][i].view(1, -1)]  # All shaped (1, k)
            dists = torch.cat(tuple(dists), dim=0)  # shape: (bs, k)

            # Store the indices of the leaves with the highest path probability
            info["out_leaf_ix"] = [leaves[i.item()].index for i in ix]

            return dists, info
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[2]:  # Greedy
            # At every decision node, the child with highest probability will be chosen
            batch_size = xs.size(0)
            # Set the threshold for when either child is more likely
            threshold = 0.5 if not self._log_probabilities else np.log(0.5)
            # Keep track of the routes taken for each of the items in the batch
            routing = [[] for _ in range(batch_size)]
            # Traverse the tree for all items
            # Keep track of all nodes encountered
            for i in range(batch_size):
                node = self._root
                while node in self.internal_nodes:
                    routing[i] += [node]
                    if attr[node, "p_right"][i].item() > threshold:
                        node = node.right
                    else:
                        node = node.left
                routing[i] += [node]

            # Obtain output distributions of each leaf
            # Each selected leaf is at the end of a path stored in the `routing` variable
            dists = [attr[path[-1], "ds"][0] for path in routing]
            # Concatenate the dists in a new batch dimension
            dists = torch.cat([dist.unsqueeze(0) for dist in dists], dim=0).to(
                device=xs.device
            )

            # Store info
            info["out_leaf_ix"] = [path[-1].index for path in routing]

            return dists, info
        raise Exception("Sampling strategy not recognized!")

    def forward_partial(self, xs: torch.Tensor) -> tuple:
        features = self.extract_features(xs)
        # (batch_size, num_prototypes, W, H)
        distances = self.prototype_layer(features)
        return features, distances, copy(self.out_map)

    @property
    def depth(self) -> int:
        return self.root.depth

    @property
    def size(self) -> int:
        return self.root.size

    # TODO: don't descendants already contain leaves? Seems like this method is redundant.
    @property
    def all_nodes(self) -> set:
        return self.root.descendants

    @property
    def node_by_index(self) -> dict:
        return self.root.node_by_index

    # TODO: this shouldn't be a property (among many other properties)
    @property
    def node_depths(self) -> dict[Node, int]:
        node2depth = {}

        def assign_depths(node, cur_depth):
            node2depth[node] = cur_depth
            for child in node.child_nodes:
                assign_depths(child, cur_depth + 1)

        assign_depths(self.root, 0)
        return node2depth

    @property
    def internal_nodes(self) -> set:
        return self.root.descendant_internal_nodes

    @property
    def leaves(self) -> set:
        return self.root.leaves

    @property
    def num_internal_nodes(self) -> int:
        return self.root.num_internal_nodes

    @property
    def num_leaves(self) -> int:
        return self.root.num_leaves

    def save(self, basedir: str, file_name: str = "model.pth"):
        self.eval()
        basedir = Path(basedir)
        basedir.mkdir(parents=True, exist_ok=True)
        torch.save(self, basedir / file_name)

    def save_state(
        self,
        basedir: str,
        state_file_name="model_state.pth",
        pickle_file_name="tree.pkl",
    ):
        basedir = Path(basedir)
        basedir.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), basedir / state_file_name)
        # TODO: what's up with this?
        with open(basedir / pickle_file_name, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + "/model.pth")

    # TODO: move whole parent logic to tree (i.e. to node and root node in particular) instead of this class
    def _set_parents(self) -> None:
        self.node2parent.clear()
        self.node2parent[self.root] = None

        def _set_parents_recursively(node: Node):
            if isinstance(node, InternalNode):
                self.node2parent[node.right] = node
                self.node2parent[node.left] = node
                _set_parents_recursively(node.right)
                _set_parents_recursively(node.left)
                return
            if isinstance(node, Leaf):
                return  # Nothing to do here!
            raise Exception("Unrecognized node type!")

        # Set all parents by traversing the tree starting from the root
        _set_parents_recursively(self.root)

    def path_to(self, node: Node):
        assert node in self.leaves or node in self.internal_nodes
        path = [node]
        while isinstance(self.node2parent[node], Node):
            node = self.node2parent[node]
            path = [node] + path
        return path
